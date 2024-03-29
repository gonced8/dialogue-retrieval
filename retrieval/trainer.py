from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
import torch

from model.save_examples import SaveExamples
from utils import none_or_str


class Trainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument("--data_name", type=str)
        parser.add_argument("--model_name", type=str)
        parser.add_argument(
            "--mode",
            type=str,
            default="test",
            choices=["train", "test", "validate"],
        )
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--accumulate_grad_batches", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=1)
        parser.add_argument("--num_sanity_val_steps", type=int, default=0)
        parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
        parser.add_argument("--val_check_interval", type=float, default=1.0)
        parser.add_argument("--monitor", type=str, default="val_loss")
        parser.add_argument("--monitor_mode", type=str, default="min")
        parser.add_argument("--patience", type=int, default=3)
        parser.add_argument("--log_every_n_steps", type=int, default=1)
        parser.add_argument("--save_examples", action="store_true")
        parser.add_argument(
            "--ckpt_path", type=none_or_str, default=None, help="Checkpoint path"
        )
        parser.add_argument("--enable_checkpointing", action="store_true")
        parser.add_argument("--default_root_dir", type=str, default="checkpoints")
        parser.add_argument("--fast_dev_run", action="store_true")

        # parent_parser = pl.Trainer.add_argparse_args(parent_parser)

        return parent_parser

    @staticmethod
    def from_argparse_args(args):
        if args.model_name is None:
            args.model_name = ""
        if args.data_name is None:
            args.data_name = ""

        # Get folder name
        name = f"{args.model_name.lower().replace('_', '')}_{args.data_name.lower().replace('_', '')}"

        # Callbacks and Logger
        if args.enable_checkpointing and not args.fast_dev_run:
            callbacks = []

            callbacks += [
                ModelCheckpoint(
                    filename="checkpoint_epoch={epoch:02d}-rougeL={rougeL:.4f}",
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    save_last=True,
                )
            ]

            callbacks += [
                EarlyStopping(
                    monitor=args.monitor,
                    mode=args.monitor_mode,
                    min_delta=1e-4,
                    patience=args.patience,
                    strict=True,
                )
            ]

            if args.lr == 0:
                args.lr = None
            else:
                callbacks += [LearningRateMonitor()]

            if args.save_examples:
                callbacks += [SaveExamples()]

            version = "version_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")
            print(f"Version: {version}")

            logger = TensorBoardLogger(
                save_dir="checkpoints",
                name=name,
                version=version,
            )
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
