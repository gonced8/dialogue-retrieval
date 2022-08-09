import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import torch

from model import SaveExamples


def none_or_str(value):
    if value == "None":
        return None
    return value


class Trainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument(
            "--mode",
            type=str,
            default="test",
            choices=["train", "test", "validate"],
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--monitor", type=str, default="val_loss")
        parser.add_argument("--monitor_mode", type=str, default="min")
        parser.add_argument("--save_val", action="store_true")
        parser.add_argument("--save_test", action="store_true")
        parser.add_argument(
            "--ckpt_path", type=none_or_str, default=None, help="Checkpoint path"
        )

        parent_parser = pl.Trainer.add_argparse_args(parent_parser)

        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        # Get folder name
        name = f"{args.model_name.lower()}_{args.data_name.lower()}"

        # Callbacks and Logger
        if args.enable_checkpointing and not args.fast_dev_run:
            callbacks = []

            callbacks += [
                ModelCheckpoint(
                    filename="best", monitor=args.monitor, mode="min", save_last=True
                )
            ]

            callbacks += [
                EarlyStopping(
                    monitor=args.monitor,
                    mode="min",
                    min_delta=1e-4,
                    patience=5,
                    strict=True,
                )
            ]

            if args.lr == 0:
                args.lr = None
            else:
                callbacks += [LearningRateMonitor()]

            if args.save_val:
                callbacks += [SaveExamples()]

            logger = TensorBoardLogger(save_dir="checkpoints", name=name)
        else:
            callbacks = None
            logger = False

        gpus = torch.cuda.device_count()
        accelerator = "ddp" if gpus > 1 else "gpu" if gpus == 1 else None
        plugins = DDPPlugin(find_unused_parameters=False) if gpus > 1 else None

        return pl.Trainer.from_argparse_args(
            args,
            accelerator=accelerator,
            gpus=gpus,
            plugins=plugins,
            callbacks=callbacks,
            logger=logger,
        )
