import json
import math
import os
import random

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .lcs import lcs_similarity


class MultiWOZ(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.seed = args.seed
        self.data_dir = args.data_dir
        self.total_batch_size = args.total_batch_size
        self.total_val_batch_size = args.total_val_batch_size
        self.total_test_batch_size = args.total_test_batch_size
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.annotations = ["domains", "acts", "slots", "values"]
        self.tokenizer = None

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
        if stage in (None, "fit", "train"):
            self.train_dataset = MultiWOZDataset(
                self.datasets_filenames["train"],
                total_batch_size=self.total_batch_size,
                seed=self.seed + 0,
            )

        if stage in (None, "fit", "val"):
            self.val_dataset = MultiWOZDataset(
                self.datasets_filenames["val"],
                dialogues_per_sample=self.hparams.mrr_total + 1,
                total_batch_size=self.total_val_batch_size,
                seed=self.seed + 0,
            )

        if stage in (None, "test"):
            self.val_dataset = MultiWOZDataset(
                self.datasets_filenames["test"],
                total_batch_size=total_test_batch_size,
                seed=self.seed + 0,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_fit,
            shuffle=True,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn_fit,
            shuffle=False,
            pin_memory=bool(torch.cuda.device_count()),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
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
            [MultiWOZDataset.get_conversation(d) for d in sample["dialogues"]]
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
            [reference for _, *reference in texts],
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
        parser = parent_parser.add_argument_group("DataModule: MultiWOZ")
        parser.add_argument("--data_name", type=str, default="multiwoz")
        parser.add_argument(
            "--data_dir", type=str, default="../data/multiwoz/processed/"
        )
        parser.add_argument("--total_batch_size", type=int, default=100000)
        parser.add_argument("--total_val_batch_size", type=int, default=1000)
        parser.add_argument("--total_test_batch_size", type=int, default=1000)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--val_batch_size", type=int, default=1)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--mrr_total", type=int, default=10)
        parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
        return parent_parser


class RandomCombinationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, r, total_batch_size=1000, seed=None):
        self.dataset = dataset
        self.r = r
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


class MultiWOZDataset(RandomCombinationDataset):
    def __init__(
        self,
        filename,
        annotations=["domains", "acts", "slots", "values"],
        dialogues_per_sample=2,
        total_batch_size=1000,
        seed=None,
    ):
        self.annotations = annotations
        self.segments = None

        with open(filename, "r") as f:
            dataset = json.load(f)

        # Initialize dataset and randomize
        super().__init__(dataset, dialogues_per_sample, total_batch_size, seed)

    def __len__(self):
        return self.total_batch_size

    def __getitem__(self, idx):
        # Gets dialogue ids
        d_ids = super().__getitem__(idx)
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

        return {"id": sample_id, "dialogues": dialogues, "similarity": similarity}

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

    @staticmethod
    def get_conversation(dialogue, speaker=True):
        return "\n".join(
            (f"{turn['speaker']+': ':>8}" if speaker else "") + turn["utterance"]
            for turn in dialogue
        )

    @staticmethod
    def get_sequence(dialogue, annotations, flatten=False):
        sequence = []

        # Loop through turns
        for turn in dialogue:
            subsequence = []

            # Loop through dialogue acts
            for dialogue_act, slots_dict in turn["dialogue_acts"].items():
                domain, dialogue_act = dialogue_act.split("-")

                # Special case where there is no slots/values or we don't want them
                if not slots_dict or not (
                    "slots" in annotations or "values" in annotations
                ):
                    slots_dict = {None: None}

                # Loop through slots and values
                for slot, value in slots_dict.items():
                    element = []

                    if "domains" in annotations:
                        element.append(domain)
                    if "acts" in annotations:
                        element.append(dialogue_act)
                    if "slots" in annotations and slot is not None:
                        element.append(slot)
                    if "values" in annotations and value is not None:
                        element.append(value)

                    if element:
                        subsequence.append(tuple(element))

            if subsequence:
                sequence.append(subsequence)

        # Flatten sequence
        if flatten == "concatenate":
            sequence = [
                "".join(x.title().replace(" ", "") for x in element)
                for subsequence in sequence
                for element in subsequence
            ]
        elif flatten:
            sequence = [
                x
                for subsequence in sequence
                for element in subsequence
                for x in element
            ]

        return sequence


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZ.add_argparse_args(parser)
    parser.add_argument("--mode", type=str, default="example")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = MultiWOZ(args)
    data.prepare_data()
    data.setup("fit")

    if args.mode == "example":
        dataset = data.train_dataset
        print(*dataset[0].items(), sep="\n")
        print(dataset.segments[0])

    if args.mode == "validate":
        dataset = data.val_dataset
        print(dataset[0]["id"], dataset[0]["similarity"], sep="\n")
