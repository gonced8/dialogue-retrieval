import json
import math
import os
import random

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class MultiWOZ(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.seed = args.seed
        self.data_dir = args.data_dir
        self.total_batch_size = args.total_batch_size
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def prepare_data(self):
        # Get datasets filenames and check if files exist
        self.datasets_filenames = {
            mode: os.path.join(self.data_dir, f"{mode}.json")
            for mode in ["train", "val", "test"]
        }

        self.datasets_filenames = {
            mode: filename if os.path.isfile(filename) else None
            for mode, filename in self.datasets_filenames.items()
        }

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_dataset = MultiWOZDataset(
                self.datasets_filenames["train"],
                total_batch_size=self.total_batch_size,
                seed=self.seed + 0,
            )
            self.val_dataset = MultiWOZDataset(
                self.datasets_filenames["val"],
                total_batch_size=self.total_batch_size,
                seed=self.seed + 0,
            )

        if stage in (None, "test"):
            self.val_dataset = MultiWOZDataset(
                self.datasets_filenames["test"],
                total_batch_size=total_batch_size,
                seed=self.seed + 0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("DataModule: MultiWOZ")
        parser.add_argument("--data_name", type=str, default="multiwoz")
        parser.add_argument(
            "--data_dir", type=str, default="../data/multiwoz/processed/"
        )
        parser.add_argument("--total_batch_size", type=int, default=100000)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
        return parent_parser


class RandomPairsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, total_batch_size=1000, seed=None):
        self.dataset = dataset
        self.total_batch_size = total_batch_size
        self.samples = None
        self.randomize(seed)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return self.total_batch_size

    def total_size(self):
        return math.comb(len(self.dataset), 2)

    def randomize(self, seed=None):
        self.samples = tuple(
            self.random_combinations(
                self.dataset, 2, self.total_batch_size, repeat=True, seed=seed
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


class MultiWOZDataset(RandomPairsDataset):
    def __init__(self, filename, total_batch_size=1000, seed=None):
        with open(filename, "r") as f:
            dataset = json.load(f)

        # Initialize dataset and randomize
        super().__init__(dataset, total_batch_size, seed)

    def __getitem__(self, idx):
        # Gets dialogues from dialogue ids
        pair = super().__getitem__(idx)
        pair = {d_id: self.dataset[d_id] for d_id in pair}

        # Cuts dialogue according to randomized segment size
        pair = {
            d_id: dialogue[start:end]
            for (d_id, dialogue), (start, end) in zip(pair.items(), self.segments[idx])
        }

        # TODO extract ids, dialogues, extract sequences, measure similarity, convert to InputExamples
        # TODO avoid repeating this computations

        return pair

    def randomize(self, seed=None):
        super().randomize(seed)
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
    parser = MultiWOZ.add_argparse_args(parser)
    parser.add_argument("--seed", type=str, default=42, help="seed")

    args = parser.parse_args()
    data = MultiWOZ(args)
    data.prepare_data()
    data.setup("fit")

    dataset = data.train_dataset
    print(*dataset[0].items(), sep="\n")
    print(dataset.segments[0])
