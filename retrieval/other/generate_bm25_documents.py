from argparse import ArgumentParser
import json
from pathlib import Path

from tqdm import tqdm


def generate_documents(args):
    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Get documents
    conversations = [
        {
            "id": sample["id"],
            "contents": sample["text"].rsplit("\n", 1)[0],
            "answer": sample["text"].rsplit("\n", 1)[1],
        }
        for sample in dataset
    ]

    index_directory = Path(args.index_directory) / "collection"
    index_directory.mkdir(parents=True, exist_ok=True)

    # Save documents
    for sample in tqdm(conversations, desc="Saving documents"):
        with open(index_directory / f"{sample['id']}.json", "w") as f:
            json.dump(sample, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="../data/multiwoz/processed2/test.json"
    )
    parser.add_argument(
        "--index_directory", type=str, default="data/multiwoz/index/test_bm25"
    )
    args = parser.parse_args()

    # Generate documents for BM25 index
    generate_documents(args)
