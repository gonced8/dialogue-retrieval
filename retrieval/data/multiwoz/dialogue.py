from itertools import islice
import json
import os
import random

import pytorch_lightning as pl
import torch

from utils import get_text


class MultiWOZDialogueDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.annotations = ["domains", "acts", "slots", "values"]
        self.tokenizer = None

    def setup(self, stage=None):
        # Load train data
        with open(self.hparams.train_data, "r") as f:
            self.train_data = json.load(f)
            train_data = {
                sample["id"]: {k: v for k, v in sample.items() if k != "id"}
                for sample in self.train_data
            }

        if stage in (None, "fit", "train"):
            # Load dataset (containing future triplets)
            with open(self.hparams.train_dataset, "r") as f:
                self.train_dataset = json.load(f)

            # Get triplets
            self.train_dataset = [
                self.get_triplet(
                    anchor,
                    candidates,
                    self.hparams.heuristic,
                    self.hparams.negative_st_best,
                )
                for anchor, candidates in self.train_dataset.items()
            ]

            self.train_dataset = [
                {t: {"id": d_id, **train_data[d_id]} for t, d_id in sample.items()}
                for sample in self.train_dataset
            ]

        if stage in (None, "fit", "validate"):
            # Load validation dataset
            with open(self.hparams.val_data, "r") as f:
                self.val_dataset = json.load(f)

        if stage in (None, "test"):
            # Load test dataset
            with open(self.hparams.test_data, "r") as f:
                self.test_dataset = json.load(f)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_fit,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_test,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_test,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def index_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.hparams.index_batch_size,
            num_workers=self.hparams.num_workers,
            # collate_fn=self.collate_fn_test,
            collate_fn=self.collate_fn_index,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def collate_fn_fit(self, batch):
        # Get ids
        ids = [
            sample["anchor"]["id"]
            + "_"
            + sample["positive"]["id"]
            + "_"
            + sample["negative"]["id"]
            for sample in batch
        ]

        # Get anchor, positive, negative texts
        texts = {}
        for k in batch[0].keys():
            texts[k] = [
                # get_text(sample[k], "context")
                get_text(sample[k], "context" if k == "anchor" else "answer")
                for sample in batch
            ]

        # Tokenize and convert to tensors
        if self.tokenizer is not None:
            tokenized = {}
            for k, text_samples in texts.items():
                tokenized[k] = self.tokenizer(
                    text_samples,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
        else:
            tokenized = None

        return {
            "ids": ids,
            "anchor": tokenized["anchor"],
            "positive": tokenized["positive"],
            "negative": tokenized["negative"],
        }

    def collate_fn_test(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        contexts = [sample["text"].rsplit("\n", 1)[0] for sample in batch]
        answers = [sample["text"].rsplit("\n", 1)[1] for sample in batch]

        # Tokenize and convert to tensors
        if self.tokenizer is not None:
            context_tokenized = self.tokenizer(
                contexts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            context_tokenized = None

        return {
            "ids": ids,
            "contexts": contexts,
            "answers": answers,
            "context_tokenized": context_tokenized,
        }

    def collate_fn_index(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        contexts = [sample["text"].rsplit("\n", 1)[0] for sample in batch]
        answers = [sample["text"].rsplit("\n", 1)[1] for sample in batch]

        # Tokenize and convert to tensors
        if self.tokenizer is not None:
            tokenized = self.tokenizer(
                # context,
                answers,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            tokenized = None

        return {
            "ids": ids,
            "contexts": contexts,
            "answers": answers,
            # "context_tokenized": tokenized,
            "answer_tokenized": tokenized,
        }

    @staticmethod
    def get_triplet(anchor, candidates, heuristic, negative_st_best):
        # Positive example
        positive = max(
            candidates,
            key=lambda d_id: candidates[d_id][heuristic],
        )

        # Negative example
        if not negative_st_best:
            negative = min(
                candidates,
                key=lambda d_id: candidates[d_id][heuristic],
            )
        else:
            ids = list(candidates.keys())
            negative = ids[0] if positive != ids[0] else ids[1]

        return {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule: MultiWOZDialogue")
        parser.add_argument("--train_data", type=str)
        parser.add_argument("--val_data", type=str)
        parser.add_argument("--test_data", type=str)
        parser.add_argument("--train_dataset", type=str)
        parser.add_argument("--heuristic", type=str)
        parser.add_argument("--negative_st_best", action="store_true")
        parser.add_argument("--train_batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=64)
        parser.add_argument("--test_batch_size", type=int, default=64)
        parser.add_argument("--index_batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
        return parent_parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZDialogueDataModule.add_argparse_args(parser)
    parser.add_argument("--mode", type=str, default="example")
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(test_batch_size=2)
    args = parser.parse_args()

    data = MultiWOZDialogueDataModule(args)
    data.prepare_data()
    data.setup("test")

    if args.mode == "example":
        batch = next(iter(data.test_dataloader()))
        print(batch)
