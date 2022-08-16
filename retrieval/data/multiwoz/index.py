from argparse import ArgumentParser
import json
import os
from pathlib import Path
import random
import shutil

from autofaiss import build_index
from faiss import read_index
import numpy as np
import psutil
from pytorch_lightning import seed_everything
import torch
from tqdm import tqdm

from data.multiwoz import MultiWOZ
from model.retriever import Retriever


def process_dataset(dataset, seed=None):
    random.seed(seed)
    new_dataset = []

    for d_id, dialogue in tqdm(dataset.items(), desc="Processing the dataset"):
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
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        contexts = [MultiWOZ.get_conversation(sample["context"]) for sample in batch]
        answers = [MultiWOZ.get_conversation(sample["answer"]) for sample in batch]

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


def compute_embeddings(args):
    # Initialize embeddings directory
    embeddings_dir = Path(args.index_directory) / "embeddings"
    if embeddings_dir.exists():
        print(f"Directory {embeddings_dir} already exists and contains:")
        print(*sorted(list(embeddings_dir.iterdir())), sep="\n")
        option = input("Do you want to recompute the embeddings? [y/N] ")
        if option == "" or option.lower() == "n":
            return
        else:
            shutil.rmtree(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(args.filename, "r") as f:
        dataset = json.load(f)

    # Process dataset
    new_dataset = process_dataset(dataset)

    # Load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.ckpt_path is not None:
        model = Retriever.load_from_checkpoint(args.ckpt_path, args=args)
        print(f"Loaded model from checkpoint: {args.ckpt_path}")
    else:
        model = Retriever(args)

    model.to(device)
    model.eval()

    # Initialize Dataloader
    dataloader = torch.utils.data.DataLoader(
        new_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=CollateFn(model.tokenizer),
        shuffle=False,
        pin_memory=bool(torch.cuda.device_count()),
    )

    # Encode dataset
    batch_ids = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Calculating embeddings")
        ):
            # Get input_ids and attention_mask into device
            x = {k: v.to(device) for k, v in batch["context_tokenized"].items()}

            # Compute dialogue embeddings
            embeddings = model(x)

            # Save embeddings to disk
            with open(embeddings_dir / f"{batch_idx}.npy", "wb") as f:
                np.save(f, embeddings.cpu().numpy())

            # Update batch to dialogue ids correspondence
            batch_ids[batch_idx] = batch["ids"]

    # Save batch to dialogue ids correspondence
    with open(embeddings_dir / "batch_ids.json", "w") as f:
        json.dump(batch_ids, f, indent=4)

    print(f"Saved embeddings to {embeddings_dir}")


def generate_index(args):
    # Check if index already exists
    index_directory = Path(args.index_directory)
    if (index_directory / "index.bin").exists():
        print(f"Index at {index_directory} already exists.")
        option = input("Do you want to regenerate the index? [y/N] ")
        if option == "" or option.lower() == "n":
            return

    # Get current available memory in GB
    current_memory_available = f"{psutil.virtual_memory().available * 2**-30:.0f}G"

    build_index(
        embeddings=str(index_directory / "embeddings"),
        index_path=str(index_directory / "index.bin"),
        index_infos_path=str(index_directory / "info.json"),
        max_index_memory_usage="32G",
        current_memory_available=current_memory_available,
        metric_type="ip",
    )

    # Generate file with correspondence between ids
    with open(index_directory / "embeddings" / "batch_ids.json", "r") as f:
        batch_ids = json.load(f)

    ids_labels = dict(enumerate([l for labels in batch_ids.values() for l in labels]))

    with open(index_directory / "ids_labels.json", "w") as f:
        json.dump(ids_labels, f, indent=4)

    print(f"Built index to {index_directory}")


def test(args):
    # Load dataset
    with open(args.filename, "r") as f:
        dataset = json.load(f)

    # Process dataset
    new_dataset = process_dataset(dataset)

    # Read index
    index_directory = Path(args.index_directory)
    index = read_index(str(index_directory / "index.bin"))
    with open(index_directory / "ids_labels.json", "r") as f:
        ids_labels = list(json.load(f).values())

    # Load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.ckpt_path is not None:
        model = Retriever.load_from_checkpoint(args.ckpt_path, args=args)
    else:
        model = Retriever(args)

    model.to(device)
    model.eval()

    # Tokenize example dialogue
    example = """  USER: What time does the train depart?
SYSTEM: It departs at 9 am.
  USER: How much does it cost?"""

    x = model.tokenizer(
        [example],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    # Get input_ids and attention_mask into device
    x = {k: v.to(device) for k, v in x.items()}

    # Compute dialogue embeddings
    with torch.no_grad():
        embeddings = model(x)

    # Normalize embeddings because of cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    # Search
    k = 5
    distances, indices = index.search(embeddings.cpu().numpy(), k)

    results = [
        f"""{idx}: {ids_labels[idx]}\tscore: {distance}
{MultiWOZ.get_conversation(new_dataset[idx]["context"])}
{MultiWOZ.get_conversation(new_dataset[idx]["answer"])}"""
        for idx, distance in zip(indices[0], distances[0])
    ]

    print("Example:", example, "", sep="\n")
    print("Results:", *results, sep="\n\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="../data/multiwoz/processed/test.json"
    )
    parser.add_argument(
        "--original_model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument("--index_directory", type=str, default="data/multiwoz/index/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed, workers=True)

    compute_embeddings(args)
    generate_index(args)
    test(args)
