from argparse import ArgumentParser
import json
from pathlib import Path

from pytorch_lightning import seed_everything
from tqdm import tqdm

from data.multiwoz.single import MultiWOZSingleDataModule


def generate_documents(dataset, index_directory):
    # Get documents
    conversations = [
        {
            "id": sample["id"],
            "contents": sample["context"],
            "answer": sample["answer"],
        }
        for sample in dataset
    ]

    index_directory = Path(index_directory) / "collection"
    index_directory.mkdir(parents=True, exist_ok=True)

    # Save documents
    for sample in tqdm(conversations, desc="Saving documents"):
        with open(index_directory / f"{sample['id']}.json", "w") as f:
            json.dump(sample, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = MultiWOZSingleDataModule.add_argparse_args(parser)
    parser.add_argument("--index_directory", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed, workers=True)

    # Load test dataset
    data = MultiWOZSingleDataModule(args)
    data.prepare_data()
    data.setup("test")
    dataset = data.test_dataset

    # Generate documents for BM25 index
    generate_documents(dataset, args.index_directory)
