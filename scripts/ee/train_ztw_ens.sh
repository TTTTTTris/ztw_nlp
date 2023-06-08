#!/bin/bash

task=$1
seed=42
epochs=50
# patience=5
# batch_size=64

# ens_lr=2e-4
# lr=2e-3

# ztw_ens_dir="models/ztw_${task}_${seed}_${lr}_${ens_lr}"
ztw_ens_dir="models/ztw_${task}"
export CUDA_VISIBLE_DEVICES=0
python src/train.py \
  --ee_model "ztw" \
  --tags "${task}_final" \
  --task $task \
  --data_path "/home/fangchao/tianjiayi/glue_data/${task}" \
  --model_path "/home/home/fangchao/tjy/bit/outputs_double/${task}/W1A1/kd_joint" \
  --output_dir $ztw_ens_dir \
  --seed $seed \
  --num_workers 100 \
  --evaluate_ee_thresholds > nohup_${task}.out 2>&1 &
  # --ensembling_lr $ens_lr \
  # --max_epochs $epochs \
  # --val_batch_size $batch_size \
  # --lr $lr \
  # --patience $patience \
  # --ensembling \
  # --ensembling_epochs $epochs \
# rm -rf $ztw_dir