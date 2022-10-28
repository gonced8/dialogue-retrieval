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
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from data.multiwoz import MultiWOZ


def generate_dataset(args):
    dataset = MultiWOZ(args.filename)
    new_dataset = []
    exclude_indices = {}

    for d_id, dialogue in tqdm(
        dataset.items(), desc="Generating samples from dialogues"
    ):
        size = len(new_dataset)
        sequence = MultiWOZ.get_sequence(dialogue)

        for i in range(2, len(sequence) + 1, 2):
            start = max(0, i - args.max_nturns)

            sample_id = f"{d_id}_{start}-{i}"
            text = MultiWOZ.get_conversation(dialogue[start:i])
            annotations = sequence[start:i]

            sample = {"id": sample_id, "text": text, "annotations": annotations}
            new_dataset.append(sample)

        exclude_indices[d_id] = list(range(size, size + len(sequence) // 2))

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save new dataset
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(new_dataset, f, indent=4)

    with open(output_dir / "exclude_indices.json", "w") as f:
        json.dump(exclude_indices, f, indent=4)

    print(f"Saved new dataset to directory {args.output}")
    print(f"New dataset contains {len(new_dataset)} samples.")

    return new_dataset, exclude_indices


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Get ids
        ids = [sample["id"] for sample in batch]

        # Get data
        texts = [sample["text"].split("\n") for sample in batch]
        contexts = ["\n".join(text[:-1]) for text in texts]
        # answers = [text[-1] for text in texts]

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
            # "answers": answers,
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

    # Load dataset and exclude indices
    output_dir = Path(args.output)

    with open(output_dir / "dataset.json", "r") as f:
        dataset = json.load(f)

    # Load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SentenceTransformer(args.model_name)
    model.to(device)
    model.eval()

    # Initialize Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
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
            embeddings = model(x)["sentence_embedding"]

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


def retrieve(args):
    # Load dataset and exclude indices
    output_dir = Path(args.output)

    with open(output_dir / "dataset.json", "r") as f:
        dataset = json.load(f)

    with open(output_dir / "exclude_indices.json", "r") as f:
        exclude_indices = json.load(f)

    # Load index
    index_directory = Path(args.index_directory)
    index = read_index(str(index_directory / "index.bin"))
    with open(index_directory / "ids_labels.json", "r") as f:
        ids_labels = list(json.load(f).values())

    # Load model
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SentenceTransformer(args.model_name)
    model.to(device)
    model.eval()

    # Initialize Dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=CollateFn(model.tokenizer),
        shuffle=False,
        pin_memory=bool(torch.cuda.device_count()),
    )

    # Get maximum number of sub-dialogues per dialogue
    max_n_subdialogues = max(len(indices) for indices in exclude_indices.values())

    # Tokenize, encode, and retrieve
    results = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc="Retrieving using Sentence Transformer")
        ):
            # Get input_ids and attention_mask into device
            x = {k: v.to(device) for k, v in batch["context_tokenized"].items()}

            # Compute dialogue embeddings
            embeddings = model(x)["sentence_embedding"]

            # Retrieve
            distances, indices = index.search(
                embeddings.cpu().numpy(), args.k + max_n_subdialogues
            )

            # Filter sub-dialogues hits from the same original dialogue
            for sample_id, sample_ids, sample_distances in zip(
                batch["ids"], indices, distances
            ):
                full_id = sample_id.split("_")[0]
                exclude = exclude_indices[full_id]
                sample_results = []

                for hit_id, hit_distance in zip(sample_ids, sample_distances):
                    if hit_id not in exclude:
                        hit_label = ids_labels[hit_id]
                        hit_distance = str(hit_distance)
                        sample_results.append((hit_label, hit_distance))

                results[sample_id] = sample_results

    with open(args.results, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="../data/multiwoz/processed/train.json"
    )
    parser.add_argument("--output", type=str, default="../data/multiwoz/processed2")
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/afll-mpnet-base-v2",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_nturns", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
    parser.add_argument(
        "--index_directory", type=str, default="data/multiwoz/index/new_dataset"
    )
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed, workers=True)

    # Run
    # TODO: only generate dataset if necessary
    # generate_dataset(args)
    compute_embeddings(args)
    generate_index(args)
    retrieve(args)
