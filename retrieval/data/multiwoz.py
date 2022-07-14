import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from lcs import *

SAVE = True
SHOW = False


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
        # Gets dialogue ids
        d_ids = super().__getitem__(idx)

        # Get dialogues and cut them according to randomized segment size
        dialogues = [
            self.dataset[d_id][start:end]
            for d_id, (start, end) in zip(d_ids, self.segments[idx])
        ]

        # TODO extract ids, dialogues, extract sequences, measure similarity, convert to InputExamples
        # TODO avoid repeating this computations

        return {"d_ids": d_ids, "dialogues": dialogues}

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


def scatter_hist(x, y, xlabel=None, ylabel=None):
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=10, alpha=0.2)

    # now determine nice limits by hand:
    binwidth = 0.02
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = np.ceil(xymax / binwidth) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation="horizontal")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZ.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument(
        "--mode", type=str, default="example", choices=["example", "stats"]
    )
    args = parser.parse_args()

    data = MultiWOZ(args)
    data.prepare_data()
    data.setup("fit")

    if args.mode == "example":
        dataset = data.train_dataset
        print(*dataset[0].items(), sep="\n")
        print(dataset.segments[0])

    elif args.mode == "stats":
        dataset = data.val_dataset
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        dlengths1 = []
        dlengths2 = []
        dlengths3 = []
        similarities = []

        for sample in dataset:
            d_ids = sample["d_ids"]
            dialogues = sample["dialogues"]
            annotations = ["domains", "acts", "slots", "values"]

            # Get conversations, tokenize, and calculate length difference
            conversations = [MultiWOZDataset.get_conversation(d) for d in dialogues]
            l1, l2 = [len(tokenizer.tokenize(c)) for c in conversations]
            dlength1 = abs(l1 - l2) / (l1 + l2)

            # Get sequences, calculate length difference, and compute similarity
            sequences = [
                MultiWOZDataset.get_sequence(d, annotations, flatten=True)
                for d in dialogues
            ]
            l3, l4 = [len(s) for s in sequences]
            dlength2 = abs(l3 - l4) / (l3 + l4)
            similarity = lcs_similarity(*sequences)

            # Calculate difference of number of turns
            l5, l6 = [len(d) for d in dialogues]
            dlength3 = abs(l5 - l6) / (l5 + l6)

            # Append values to lists
            dlengths1.append(dlength1)
            dlengths2.append(dlength2)
            dlengths3.append(dlength3)
            similarities.append(similarity)

        # Plot similarity vs length differences
        # fig, ax = plt.subplots()
        # ax.plot(dlengths1, similarities, ".", label="$\Delta$ tokens")
        # ax.plot(dlengths2, similarities, ".", label="$\Delta$ annotations")
        # ax.plot(dlengths3, similarities, ".", label="$\Delta$ turns")
        # ax.legend()
        fig1 = scatter_hist(
            dlengths1,
            similarities,
            "Normalized text length difference",
            "LCS similarity",
        )
        fig2 = scatter_hist(
            dlengths2,
            similarities,
            "Normalized number of annotations difference",
            "LCS similarity",
        )
        fig3 = scatter_hist(
            dlengths3,
            similarities,
            "Normalized number of turns difference",
            "LCS similarity",
        )

        if SAVE:
            fig1.savefig(
                "../plots/similarity_text_length_difference.png", bbox_inches="tight"
            )
            fig2.savefig(
                "../plots/similarity_number_annotations_difference.png",
                bbox_inches="tight",
            )
            fig3.savefig(
                "../plots/similarity_number_turns_difference.png", bbox_inches="tight"
            )

        if SHOW:
            plt.show()
