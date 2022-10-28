from itertools import islice
import os
import random

import pytorch_lightning as pl
import torch
from tqdm import tqdm

from data.multiwoz import MultiWOZ


class MultiWOZSingleDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.annotations = ["domains", "acts", "slots", "values"]
        self.tokenizer = None

    def prepare_data(self):
        # Get datasets filenames and check if files exist
        self.datasets_filenames = {
            mode: os.path.join(self.hparams.data_dir, f"{mode}.json")
            for mode in ["train", "val", "test"]
        }

        self.datasets_filenames = {
            mode: filename if os.path.isfile(filename) else None
            for mode, filename in self.datasets_filenames.items()
        }

    def setup(self, stage=None):
        if stage in (None, "fit", "train"):
            self.train_dataset = MultiWOZSingleDataset(
                filename=self.datasets_filenames["train"],
                max_nturns=6,
                raw=self.hparams.raw,
                seed=self.hparams.seed + 0,
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = MultiWOZSingleDataset(
                filename=self.datasets_filenames["val"],
                max_nturns=6,
                raw=self.hparams.raw,
                seed=self.hparams.seed + 0,
            )

        if stage in (None, "test"):
            self.test_dataset = MultiWOZSingleDataset(
                filename=self.datasets_filenames["test"],
                max_nturns=6,
                raw=self.hparams.raw,
                seed=self.hparams.seed + 0,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
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
            collate_fn=self.collate_fn_fit,
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

    def randomize(self, seed=None):
        self.train_dataset.randomize(seed)
        self.val_dataset.randomize(seed)

    def collate_fn_fit(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get sequences
        # include also annotation from answer, such that encoder learns to predict possible answer
        sequences = [MultiWOZ.get_sequence(sample["context"]) for sample in batch] + [
            MultiWOZ.get_sequence(sample["answer"]) for sample in batch
        ]

        # Get conversations
        contexts = [MultiWOZ.get_conversation(sample["context"]) for sample in batch]
        answers = [MultiWOZ.get_conversation(sample["answer"]) for sample in batch]

        # Convert to tensors
        contexts_tokenized = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # answers_tokenized = self.tokenizer(
        #     answers,
        #     padding=True,
        #     truncation=True,
        #     return_tensors="pt",
        # )

        return {
            "ids": ids,
            "sequences": sequences,
            "contexts": contexts,
            "answers": answers,
            "contexts_tokenized": contexts_tokenized,
            # "answers_tokenized": answers_tokenized,
        }

    def collate_fn_test(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        contexts = [sample["context"] for sample in batch]
        answers = [sample["answer"] for sample in batch]

        # Convert to tensors
        if not self.hparams.raw:
            contexts_tokenized = self.tokenizer(
                contexts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            # answers_tokenized = self.tokenizer(
            #     answers,
            #     padding=True,
            #     truncation=True,
            #     return_tensors="pt",
            # )
        else:
            contexts_tokenized = None

        return {
            "ids": ids,
            "contexts": contexts,
            "answers": answers,
            "contexts_tokenized": contexts_tokenized,
            # "answers_tokenized": answers_tokenized,
        }

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule: MultiWOZSingle")
        parser.add_argument(
            "--data_dir", type=str, default="../data/multiwoz/processed/"
        )
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
        parser.add_argument("--raw", action="store_true")
        return parent_parser


class MultiWOZSingleDataset(MultiWOZ, torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        max_nturns=6,
        raw=False,
        seed=None,
    ):
        self.max_nturns = max_nturns
        self.raw = raw
        self.segments = None

        # Initialize dataset
        MultiWOZ.__init__(self, filename=filename)
        torch.utils.data.Dataset.__init__(self)

        self.randomize(seed)

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError

        # Take care of negative indices
        idx %= self.__len__()

        # Gets dialogue id and dialogue
        d_id, dialogue = next(islice(self.dataset.items(), idx, idx + 1))

        # Get dialogues and cut them according to randomized segment size
        start, end = self.segments[idx]
        *context, answer = dialogue[start : end + 1]

        # Get conversation as text
        if not self.raw:
            context = self.get_conversation(context)
            answer = self.get_conversation(answer, speaker=False)

        return {
            "id": f"{d_id}_{start}-{end}",
            "context": context,
            "answer": answer,
        }

    def randomize(self, seed=None):
        self.segments = self.random_segments(seed)

    def random_segments(self, seed=None):
        random.seed(seed)
        segments = []

        for d_id, dialogue in tqdm(
            self.dataset.items(), desc="Generating random segments for each dialogue"
        ):
            end = random.randrange(
                1 if dialogue[1]["speaker"] == "SYSTEM" else 2,
                len(dialogue),
                2,
            )
            start = max(0, end - self.max_nturns + 1)

            segments.append((start, end))

        return segments


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZSingleDataModule.add_argparse_args(parser)
    parser.add_argument("--mode", type=str, default="example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = MultiWOZSingleDataModule(args)
    data.prepare_data()
    data.setup("test")

    if args.mode == "example":
        dataset = data.test_dataset
        print(dataset[2])
