from argparse import ArgumentParser
import json
import random
import re

from tqdm import tqdm


def get_sample(args, index_data, sample):
    # Get context
    context = sample["context"].split("\n")

    # Get knowledge from candidates
    candidates = sample["candidate"][: args.candidates]
    knowledge = [
        index_data[candidate]["text"].rsplit("\n", 1)[1] for candidate in candidates
    ]

    # Get response
    response = sample["truth_answer"]

    # Clean white spaces
    context = [re.sub(" +", " ", sentence).strip(" ") for sentence in context]
    knowledge = [re.sub(" +", " ", sentence) for sentence in knowledge]
    response = re.sub(" +", " ", response)

    return {
        "id": sample["id"],
        "context": context,
        "knowledge": knowledge,
        "response": response,
    }


def build_generate_dataset(args):
    # Load index data
    with open(args.index_dataset, "r") as f:
        index_data = json.load(f)
        index_data = {
            sample["id"]: {k: v for k, v in sample.items() if k != "id"}
            for sample in index_data
        }

    # Load retrieval results
    with open(args.retrieval_results, "r") as f:
        retrieval_results = json.load(f)[1:]  # Skip metrics

    # Get training dataset for generation
    output = [
        get_sample(args, index_data, sample) for sample in tqdm(retrieval_results)
    ]

    # Save dataset
    with open(args.dataset_output, mode="w") as f:
        json.dump({"version": args.version, "data": output}, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--index_dataset", type=str, required=True)
    parser.add_argument("--retrieval_results", type=str, required=True)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--dataset_output", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    build_generate_dataset(args)
