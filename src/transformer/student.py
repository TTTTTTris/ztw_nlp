# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# 2022.09.25 - Add elastic quantization support
#              Meta Platforms, Inc. <zechunliu@fb.com>
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys

from io import open

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from .file_utils import WEIGHTS_NAME, CONFIG_NAME
from .configuration_bert import BertConfig
from .utils_quant import QuantizeLinear, QuantizeEmbedding, act_quant_fn, AlphaInit

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "",
}

BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}
NORM = {'layer_norm': BertLayerNorm}

class channel_w(nn.Module):
    def __init__(self,out_ch):
        super(channel_w, self).__init__()
        self.w1 =torch.nn.Parameter(torch.rand(1,1,out_ch)*0.1,requires_grad=True)

    def forward(self,x):
        out = self.w1 * x
        return out
    
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings1 = QuantizeEmbedding(config.vocab_size, config.hidden_size, padding_idx=0,
                                                 clip_val=config.clip_init_val,
                                                 weight_bits=config.weight_bits,
                                                 weight_quant_method=config.weight_quant_method,
                                                 embed_layerwise=config.embed_layerwise,
                                                 learnable=config.learnable_scaling,
                                                 symmetric=config.sym_quant_qkvo)
        self.word_embeddings2 = QuantizeEmbedding(config.vocab_size, config.hidden_size, padding_idx=0,
                                                 clip_val=config.clip_init_val,
                                                 weight_bits=config.weight_bits,
                                                 weight_quant_method=config.weight_quant_method,
                                                 embed_layerwise=config.embed_layerwise,
                                                 learnable=config.learnable_scaling,
                                                 symmetric=config.sym_quant_qkvo)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids[0].size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids[0].device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids[0])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[0])

        words_embeddings1 = self.word_embeddings1(input_ids[0])
        words_embeddings2 = self.word_embeddings2(input_ids[1])
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings0 = self.token_type_embeddings(token_type_ids[0])
        token_type_embeddings1 = self.token_type_embeddings(token_type_ids[1])

        embeddings1 = words_embeddings1 + position_embeddings + token_type_embeddings0
        embeddings2 = words_embeddings2 + position_embeddings + token_type_embeddings1
        embeddings1 = self.LayerNorm1(embeddings1)
        embeddings2 = self.LayerNorm2(embeddings2)
        embeddings1 = self.dropout(embeddings1)
        embeddings2 = self.dropout(embeddings2)
        return [embeddings1, embeddings2]

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.input_bits = config.input_bits
        self.sym_quant_ffn_attn = config.sym_quant_ffn_attn
        self.sym_quant_qkvo = config.sym_quant_qkvo
        self.input_layerwise = config.input_layerwise
        self.input_quant_method = config.input_quant_method
        self.quantize_attention_probs = not config.not_quantize_attention

        self.query1 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.query2 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.key1 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                  weight_bits=config.weight_bits, input_bits=config.input_bits,
                                  weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                  weight_quant_method=config.weight_quant_method,
                                  input_quant_method=config.input_quant_method,
                                  learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.key2 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                  weight_bits=config.weight_bits, input_bits=config.input_bits,
                                  weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                  weight_quant_method=config.weight_quant_method,
                                  input_quant_method=config.input_quant_method,
                                  learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.value1 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.value2 = QuantizeLinear(config.hidden_size, self.all_head_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.move_q1 = LearnableBias(self.all_head_size)
        self.move_q2 = LearnableBias(self.all_head_size)
        self.move_k1 = LearnableBias(self.all_head_size)
        self.move_k2 = LearnableBias(self.all_head_size)
        self.move_v1 = LearnableBias(self.all_head_size)
        self.move_v2 = LearnableBias(self.all_head_size)

        if config.input_quant_method == 'uniform' and config.input_bits < 32:
            self.register_buffer('clip_query1', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_key1', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_value1', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn1', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_query2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_key2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_value2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            self.register_buffer('clip_attn2', torch.Tensor([-config.clip_init_val, config.clip_init_val]))
            if config.learnable_scaling:
                self.clip_query1 = nn.Parameter(self.clip_query1)
                self.clip_key1 = nn.Parameter(self.clip_key1)
                self.clip_value1 = nn.Parameter(self.clip_value1)
                self.clip_attn1 = nn.Parameter(self.clip_attn1)
                self.clip_query2 = nn.Parameter(self.clip_query2)
                self.clip_key2 = nn.Parameter(self.clip_key2)
                self.clip_value2 = nn.Parameter(self.clip_value2)
                self.clip_attn2 = nn.Parameter(self.clip_attn2)
        elif (config.input_quant_method == 'elastic' or config.input_quant_method == 'bwn') and config.input_bits < 32:
            self.clip_query1 = AlphaInit(torch.tensor(1.0))
            self.clip_key1 = AlphaInit(torch.tensor(1.0))
            self.clip_value1 = AlphaInit(torch.tensor(1.0))
            self.clip_attn1 = AlphaInit(torch.tensor(1.0))
            self.clip_query2 = AlphaInit(torch.tensor(1.0))
            self.clip_key2 = AlphaInit(torch.tensor(1.0))
            self.clip_value2 = AlphaInit(torch.tensor(1.0))
            self.clip_attn2 = AlphaInit(torch.tensor(1.0))
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # self.channel_w1 = channel_w(out_ch=12)
        # self.channel_w2 = channel_w(out_ch=12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, output_att=False):
        hidden_states1, hidden_states2 = hidden_states
        mixed_query_layer1 = self.query1(hidden_states1)
        mixed_query_layer2 = self.query2(hidden_states2)
        mixed_key_layer1 = self.key1(hidden_states1)
        mixed_key_layer2 = self.key2(hidden_states2)
        mixed_value_layer1 = self.value1(hidden_states1)
        mixed_value_layer2 = self.value2(hidden_states2)

        if self.input_bits < 32:
            query_layer1 = self.move_q1(mixed_query_layer1)
            query_layer2 = self.move_q2(mixed_query_layer2)
            key_layer1 = self.move_k1(mixed_key_layer1)
            key_layer2 = self.move_k2(mixed_key_layer2)
            value_layer1 = self.move_v1(mixed_value_layer1)
            value_layer2 = self.move_v2(mixed_value_layer2)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        if self.input_bits < 32:
            query_layer1 = act_quant_fn(query_layer1, self.clip_query1, self.input_bits, quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            key_layer1 = act_quant_fn(key_layer1, self.clip_key1, self.input_bits, quant_method=self.input_quant_method,
                                     symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            value_layer1 = act_quant_fn(value_layer1, self.clip_value1, self.input_bits, quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            query_layer2 = act_quant_fn(query_layer2, self.clip_query2, self.input_bits, quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            key_layer2 = act_quant_fn(key_layer2, self.clip_key2, self.input_bits, quant_method=self.input_quant_method,
                                     symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
            value_layer2 = act_quant_fn(value_layer2, self.clip_value2, self.input_bits, quant_method=self.input_quant_method,
                                       symmetric=self.sym_quant_qkvo, layerwise=self.input_layerwise)
        attention_scores1 = torch.matmul(query_layer1, key_layer1.transpose(-1, -2))
        attention_scores2 = torch.matmul(query_layer2, key_layer2.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        attention_mask0, attention_mask1 = attention_mask
        attention_scores1 = attention_scores1 + attention_mask0
        attention_scores2 = attention_scores2 + attention_mask1
        # cellular
        # resx1 = self.channel_w1(attention_scores1)
        # resx2 = self.channel_w2(attention_scores2)
        # attention_scores1 = attention_scores1 + resx2
        # attention_scores2 = attention_scores2 + resx1

        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)
        attention_probs1 = self.dropout(attention_probs1)
        attention_probs2 = self.dropout(attention_probs2)

        if self.input_bits < 32 and self.quantize_attention_probs:
            attention_probs1 = act_quant_fn(attention_probs1, self.clip_attn1, self.input_bits, quant_method=self.input_quant_method,
                                           symmetric=self.sym_quant_ffn_attn, layerwise=self.input_layerwise)
            attention_probs2 = act_quant_fn(attention_probs2, self.clip_attn2, self.input_bits, quant_method=self.input_quant_method,
                                           symmetric=self.sym_quant_ffn_attn, layerwise=self.input_layerwise)
        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape)
        context_layer2 = context_layer2.view(*new_context_layer_shape)

        return [context_layer1, context_layer2], [attention_scores1, attention_scores2]

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.channel_w1 = channel_w(out_ch=768)
        self.channel_w2 = channel_w(out_ch=768)

    def forward(self, input_tensor, attention_mask):
        self_output, layer_att = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)

        attention_output1, attention_output2 = attention_output
        resx1 = self.channel_w1(attention_output1)
        resx2 = self.channel_w2(attention_output2)
        attention_output1 += resx2.detach()
        attention_output2 += resx1.detach()

        return [attention_output1, attention_output2], layer_att


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense1 = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.dense2 = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states1, hidden_states2 = hidden_states
        if isinstance(input_tensor, list):
            input_tensor1, input_tensor2 = input_tensor
        else: 
            input_tensor1 = input_tensor
            input_tensor2 = input_tensor
        hidden_states1 = self.dense1(hidden_states1)
        hidden_states2 = self.dense2(hidden_states2)
        hidden_states1 = self.dropout(hidden_states1)
        hidden_states2 = self.dropout(hidden_states2)
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2)

        return [hidden_states1, hidden_states2]


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense1 = QuantizeLinear(config.hidden_size, config.intermediate_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.dense2 = QuantizeLinear(config.hidden_size, config.intermediate_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)

        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # self.channel_w1 = channel_w(out_ch=3072)
        # self.channel_w2 = channel_w(out_ch=3072)

    def forward(self, hidden_states):
        hidden_states1, hidden_states2 = hidden_states
        hidden_states1 = self.dense1(hidden_states1)
        hidden_states2 = self.dense2(hidden_states2)
        hidden_states1 = self.intermediate_act_fn(hidden_states1)
        hidden_states2 = self.intermediate_act_fn(hidden_states2)

        # resx1 = self.channel_w1(hidden_states1)
        # resx2 = self.channel_w2(hidden_states2)
        # hidden_states1 += resx2.detach()
        # hidden_states2 += resx1.detach()

        return [hidden_states1, hidden_states2]


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense1 = QuantizeLinear(config.intermediate_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_ffn_attn)
        self.dense2 = QuantizeLinear(config.intermediate_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_ffn_attn)
        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.channel_w1 = channel_w(out_ch=config.hidden_size)
        self.channel_w2 = channel_w(out_ch=config.hidden_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states1, hidden_states2 = hidden_states
        input_tensor1, input_tensor2 = input_tensor
        hidden_states1 = self.dense1(hidden_states1)
        hidden_states2 = self.dense2(hidden_states2)
        hidden_states1 = self.dropout(hidden_states1)
        hidden_states2 = self.dropout(hidden_states2)
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2) # 8,128,768

        resx1 = self.channel_w1(hidden_states1)
        resx2 = self.channel_w2(hidden_states2)
        hidden_states1 += resx2.detach()
        hidden_states2 += resx1.detach()

        return [hidden_states1, hidden_states2]


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, layer_att = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, layer_att # reps经过ATT+FFN


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers1 = []
        all_encoder_layers2 = []
        all_encoder_atts1 = []
        all_encoder_atts2 = []
        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att = layer_module(hidden_states, attention_mask)
            hidden_states1, hidden_states2 = hidden_states

            all_encoder_layers1.append(hidden_states1)
            all_encoder_layers2.append(hidden_states2) 
            layer_att1, layer_att2 = layer_att

            all_encoder_atts1.append(layer_att1)
            all_encoder_atts2.append(layer_att2)
        return [all_encoder_layers1, all_encoder_layers2], [all_encoder_atts1, all_encoder_atts2]

class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()
        self.dense1 = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.dense2 = QuantizeLinear(config.hidden_size, config.hidden_size, clip_val=config.clip_init_val,
                                    weight_bits=config.weight_bits, input_bits=config.input_bits,
                                    weight_layerwise=config.weight_layerwise, input_layerwise=config.input_layerwise,
                                    weight_quant_method=config.weight_quant_method,
                                    input_quant_method=config.input_quant_method,
                                    learnable=config.learnable_scaling, symmetric=config.sym_quant_qkvo)
        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        hidden_states1, hidden_states2 = hidden_states
        pooled_output1 = self.dense1(hidden_states1[-1][:, 0]) 
        pooled_output2 = self.dense2(hidden_states2[-1][:, 0]) 
        pooled_output1 = self.activation(pooled_output1)
        pooled_output2 = self.activation(pooled_output2)
        # pooled_output = self.activation(torch.cat((pooled_output1,pooled_output2)))
        # pooled_output = self.activation(pooled_output1 + pooled_output2)
        return [pooled_output1, pooled_output2]

class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_scratch(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        resolved_config_file = os.path.join(
            pretrained_model_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)

        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)
        config = kwargs.get('config', None)
        kwargs.pop('config', None)
        if config is None:
            # Load config
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            config = BertConfig.from_json_file(config_file)

        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(
                pretrained_model_name_or_path, WEIGHTS_NAME)
            logger.info("Loading model {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'

        logger.info('loading model...')
        load(model, prefix=start_prefix)
        logger.info('done!')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: output a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: output only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, output_att=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids[0])
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids[0])

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = {}
        for i in range(2):
            extended_attention_mask[i] = attention_mask[i].unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
            extended_attention_mask[i] = extended_attention_mask[i].to(
            dtype=torch.float32) # next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask[i] = (1.0 - extended_attention_mask[i]) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, layer_atts = self.encoder(embedding_output,extended_attention_mask)
        pooled_output = self.pooler(encoded_layers)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        if not output_att:
            return encoded_layers, pooled_output
        return encoded_layers, layer_atts, pooled_output

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                output_att=False,
                output_hidden=False):
        sequence_output, att_output, pooled_output = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True, output_att=True)
        pooled_output1, pooled_output2 = pooled_output
        pooled_output1 = self.dropout(pooled_output1)
        pooled_output2 = self.dropout(pooled_output2)
        logits1 = self.classifier(pooled_output1) # 考虑经过classifier后vote
        logits2 = self.classifier(pooled_output2) # 考虑经过classifier后vote

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss1 = loss_fct(logits1.view(-1, self.num_labels), labels.view(-1))
            loss2 = loss_fct(logits2.view(-1, self.num_labels), labels.view(-1))
            return [loss1,loss2], att_output, sequence_output
        else:
            return [logits1,logits2], att_output, sequence_output

class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                start_positions=None,
                end_positions=None):

        sequence_output, att_output, _ = self.bert(input_ids,token_type_ids,attention_mask)

        logits = self.qa_outputs(sequence_output[-1])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, att_output, sequence_output

        return (start_logits, end_logits), att_output, sequence_output
