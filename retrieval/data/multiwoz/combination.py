import math
import os
import random

import joblib
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import QuantileTransformer
import torch
from tqdm import tqdm

from data.multiwoz import MultiWOZ
from utils.lcs import lcs_similarity


class MultiWOZCombinationDataModule(pl.LightningDataModule):
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
        if self.hparams.transformation is not None and self.hparams.transformation:
            print(f"Loading labels transformation from {self.hparams.transformation}")
            qt = joblib.load(self.hparams.transformation)
            transformation = lambda x: qt.transform(np.array(x).reshape(-1, 1)).tolist()
        else:
            transformation = None

        if stage in (None, "fit", "train"):
            self.train_dataset = MultiWOZCombinationDataset(
                self.datasets_filenames["train"],
                dialogues_per_sample=2,
                total_batch_size=self.hparams.total_batch_size,
                transformation=transformation,
                seed=self.hparams.seed + 0,
            )

        if stage in (None, "fit", "validate"):
            self.val_dataset = MultiWOZCombinationDataset(
                self.datasets_filenames["val"],
                dialogues_per_sample=self.hparams.candidates + 1,
                total_batch_size=self.hparams.total_val_batch_size,
                transformation=transformation,
                seed=self.hparams.seed + 0,
            )

        if stage in (None, "test"):
            self.test_dataset = MultiWOZCombinationDataset(
                self.datasets_filenames["test"],
                dialogues_per_sample=self.hparams.candidates + 1,
                total_batch_size=self.hparams.total_test_batch_size,
                transformation=transformation,
                seed=self.hparams.seed + 0,
            )

    def train_dataloader(self):
        return torch.utils.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_fit,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return torch.utils.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.collate_fn_fit,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return torch.utils.DataLoader(
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

        # Get data
        texts = [
            [MultiWOZ.get_conversation(d) for d in sample["dialogues"]]
            for sample in batch
        ]

        labels = [sample["similarity"] for sample in batch]

        # Convert to tensors
        sources = self.tokenizer(
            [source for source, *_ in texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        references = self.tokenizer(
            [r for _, *references in texts for r in references],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = torch.tensor(labels).view(len(batch), -1)

        return {
            "ids": ids,
            "sources": sources,
            "references": references,
            "labels": labels,
        }

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule: MultiWOZCombination")
        parser.add_argument(
            "--data_dir", type=str, default="../data/multiwoz/processed/"
        )
        parser.add_argument("--total_batch_size", type=int, default=100000)
        parser.add_argument("--total_val_batch_size", type=int, default=1000)
        parser.add_argument("--total_test_batch_size", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--candidates", type=int, default=10)
        parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
        parser.add_argument("--transformation", type=str, default=None)
        return parent_parser


class RandomCombinationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, r, total_batch_size=1000, seed=None):
        self.dataset = dataset
        self.r = r
        self.total_batch_size = total_batch_size
        self.samples = None
        self.randomize(seed)

        super().__init__()

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.total_batch_size

    def total_size(self):
        return math.comb(len(self.dataset), 2)

    def randomize(self, seed=None):
        self.samples = tuple(
            self.random_combinations(
                self.dataset, self.r, self.total_batch_size, repeat=True, seed=seed
            )
        )

    @staticmethod
    def random_combinations(iterable, r, k=1, repeat=False, seed=None):
        """Random selection from itertools.combinations(iterable, r)
        Returns k combinations
        """
        pool = tuple(iterable)
        n = len(pool)
        random.seed(seed)

        for _ in range(k):
            if repeat:
                indices = random.choices(range(n), k=r)
            else:
                indices = random.sample(range(n), k=r)
            yield tuple(pool[i] for i in indices)


class MultiWOZCombinationDataset(MultiWOZ, RandomCombinationDataset):
    def __init__(
        self,
        filename,
        annotations=["domains", "acts", "slots", "values"],
        dialogues_per_sample=2,
        total_batch_size=1000,
        transformation=None,
        seed=None,
    ):
        self.annotations = annotations
        self.transformation = transformation
        self.segments = None

        # Initialize dataset and randomize
        MultiWOZ.__init__(self, filename)
        RandomCombinationDataset.__init__(
            self, self.dataset, dialogues_per_sample, total_batch_size, seed
        )

    def __len__(self):
        return self.total_batch_size

    def __getitem__(self, idx):
        # Gets dialogue ids
        d_ids = RandomCombinationDataset.__getitem__(idx)
        sample_id = "__".join(
            f"{d_id}_{start}-{end}"
            for d_id, (start, end) in zip(d_ids, self.segments[idx])
        )

        # Get dialogues and cut them according to randomized segment size
        dialogues = [
            self.dataset[d_id][start:end]
            for d_id, (start, end) in zip(d_ids, self.segments[idx])
        ]

        # Compute similarity
        similarity = [
            lcs_similarity(
                self.get_sequence(dialogues[0], self.annotations, True),
                self.get_sequence(d, self.annotations, True),
            )
            for d in dialogues[1:]
        ]

        # Apply label transformation
        if self.transformation is not None:
            similarity = self.transformation(similarity)

        return {"id": sample_id, "dialogues": dialogues, "similarity": similarity}

    def randomize(self, seed=None):
        RandomCombinationDataset.randomize(seed)
        self.segments = self.random_segments(seed)

    def random_segments(self, seed=None):
        random.seed(seed)
        return [
            [
                sorted(random.sample(range(0, n_turns), 2))
                for n_turns in [
                    len(self.dataset[d_id]) for d_id in pair_ids
                ]  # n_turns of each dialogue
            ]
            for pair_ids in tqdm(
                self.samples,
                desc="Generating random segments for each sample (pair of dialogues)",
            )
        ]


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZCombinationDataModule.add_argparse_args(parser)
    parser.add_argument("--mode", type=str, default="example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = MultiWOZCombinationDataModule(args)
    data.prepare_data()
    data.setup("fit")

    if args.mode == "example":
        dataset = data.train_dataset
        print(*dataset[0].items(), sep="\n")
        print(dataset.segments[0])

    if args.mode == "validate":
        dataset = data.val_dataset
        print(dataset[0]["id"], dataset[0]["similarity"], sep="\n")
