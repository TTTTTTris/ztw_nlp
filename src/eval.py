import os
import torch
import pytorch_lightning as pl
import torch.utils.data as Data
from dotenv import load_dotenv
from pytorch_lightning.loggers.wandb import WandbLogger
from utils_glue import *
from args import Config, parse_args
from consts import TASK_TO_VAL_SPLIT_NAME
# from dataset import get_data_loaders, get_validation_data_loaders_for_ee
from models.bag_model import EarlyExitModel
from utils import get_run_name, set_seed
from tokenization import BertTokenizer

load_dotenv()



def evaluate(config: Config) -> None:
    set_seed(config.seed)
    # restore the default setting if they are None
    if config.task in default_params:
        config.lr = default_params[config.task]["learning_rate"]

    if config.task in default_params:
        config.max_epochs = default_params[config.task]["num_train_epochs"]

    if config.task in default_params:
        config.train_batch_size = default_params[config.task]["batch_size"]
        config.train_batch_size = int(config.train_batch_size)

    if config.task in default_params:
        config.max_seq_length = default_params[config.task]["max_seq_length"]

    # train_data_loader, val_data_loader = get_data_loaders(config)
    processor = processors[config.task]()
    output_mode = output_modes[config.task]
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(config.model_path)
    eval_examples = processor.get_dev_examples(config.data_path)
    eval_features = convert_examples_to_features(eval_examples, label_list, config.max_seq_length, tokenizer, output_mode)
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = Data.SequentialSampler(eval_data)
    eval_dataloader = Data.DataLoader(eval_data, sampler=eval_sampler, batch_size=config.val_batch_size, num_workers=config.num_workers)

    tags = config.tags
    if tags is not None:
        tags = tags.split(",")
    logger = WandbLogger(
        entity="tris020425", # os.environ["WANDB_ENTITY"],
        project="ZTW_NLP", # os.environ["WANDB_PROJECT"],
        name=get_run_name(config),
        tags=tags,
    )
    # Create a TensorBoard logger
    # logger = TensorBoardLogger("logs/", name="ztw_RTE")
    # logger = CSVLogger("logs/", name="ztw_RTE")

    model_name = ['0.bin','1.bin']
    model = EarlyExitModel(config, model_name)
    # checkpoint_path = _eval(
    #     config=config,
    #     model=model,
    #     val_data_loader=eval_dataloader,
    #     max_epochs=config.max_epochs,
    #     logger=logger,
    #     monitor_metric=f"{TASK_TO_VAL_SPLIT_NAME[config.task]}/total_loss",
    # )
    checkpoint_path = config.output_dir/get_run_name(config)/'epoch=3-step=45484.ckpt'
    if config.evaluate_ee_thresholds:
        model = EarlyExitModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config, model_name=model_name)
        if config.limit_val_batches is not None:
            eval_data = Data.Subset(eval_data, range(config.limit_val_batches))
        eval_ee_dataloader = Data.DataLoader(eval_data, batch_size=1, num_workers=config.num_workers)
        # val_ee_data_loader = get_validation_data_loaders_for_ee(config)
        model.log_final_metrics(eval_ee_dataloader, TASK_TO_VAL_SPLIT_NAME[config.task])


def _eval(
    config: Config,
    model: pl.LightningModule,
    val_data_loader: torch.utils.data.DataLoader,
    max_epochs: int,
    logger: pl.loggers.WandbLogger,
    monitor_metric: str,
) -> str:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=config.output_dir / get_run_name(config),
        monitor=monitor_metric,
    )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelSummary(),
    ]





    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        gpus=1 if config.device == "cuda" else None,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        limit_val_batches=config.limit_val_batches,
    )
    trainer.validate(
        model=model,
        dataloaders=val_data_loader,
    )
    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    evaluate(parse_args())
