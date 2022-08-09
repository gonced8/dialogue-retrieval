from argparse import ArgumentParser
import json
import os
import random

from sentence_transformers import losses, SentenceTransformer
import torch
from tqdm import tqdm

from data.multiwoz import MultiWOZDataset


def process_dataset(dataset, seed=None):
    new_dataset = []

    for d_id, dialogue in tqdm(dataset.items(), desc="Processing the dataset..."):
        end = random.randrange(
            1 if dialogue[1]["speaker"] == "SYSTEM" else 2, len(dialogue), 2
        )
        start = random.randrange(0, end)

        new_dataset.append(
            {
                "id": f"{d_id}_{start}-{end}",
                "context": dialogue[start:end],
                "answer": dialogue[end : end + 1],
            }
        )

    return new_dataset


class CollateFn:
    def __init__(self, model_name):
        self.tokenizer = SentenceTransformer(model_name).tokenizer

    def __call__(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        contexts = [
            MultiWOZDataset.get_conversation(sample["context"]) for sample in batch
        ]
        answers = [
            MultiWOZDataset.get_conversation(sample["answer"]) for sample in batch
        ]

        # Tokenize and convert to tensors
        context_tokenized = self.tokenizer(
            contexts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "ids": ids,
            "contexts": contexts,
            "answers": answers,
            "context_tokenized": context_tokenized,
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="../data/multiwoz/processed/val.json"
    )
    parser.add_argument(
        "--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--index_folder", type=str, default="data/index/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.filename, "r") as f:
        dataset = json.load(f)

    new_dataset = process_dataset(dataset)

    dataloader = torch.utils.data.DataLoader(
        new_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=CollateFn(args.model_name),
        shuffle=False,
        pin_memory=bool(torch.cuda.device_count()),
    )

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Looping dataloader...")):
        print(batch_idx)
        print(batch)
        input()
