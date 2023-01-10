from argparse import ArgumentParser
import json
import random
import re

from tqdm import tqdm


def get_sample(args, data, index_data, anchor, candidates):
    # Get context and response
    context, response = data[anchor]["text"].rsplit("\n", 1)
    context = context.split("\n")

    # Select candidates
    if args.mode == "none":
        candidates = []
    elif args.mode == "random":
        candidates = random.sample(list(index_data.keys()), args.candidates)
    elif args.mode == "sample":
        candidates = random.sample(list(candidates.keys()), args.candidates)
    elif args.mode == "best":
        candidates = sorted(
            candidates, key=lambda d_id: candidates[d_id]["answer_rougeL"], reverse=True
        )[: args.candidates]

    # Get knowledge from candidates
    knowledge = [
        index_data[candidate]["text"].rsplit("\n", 1)[1] for candidate in candidates
    ]

    # Clean white spaces
    context = [re.sub(" +", " ", sentence).strip(" ") for sentence in context]
    knowledge = [re.sub(" +", " ", sentence) for sentence in knowledge]
    response = re.sub(" +", " ", response)

    return {
        "id": anchor,
        "context": context,
        "knowledge": knowledge,
        "response": response,
    }


def build_generate_dataset(args):
    # Load dataset
    with open(args.dataset, "r") as f:
        data = json.load(f)
        data = {
            sample["id"]: {k: v for k, v in sample.items() if k != "id"}
            for sample in data
        }

    # Load index data
    if args.index_dataset == args.dataset:
        index_data = data
    else:
        with open(args.index_dataset, "r") as f:
            index_data = json.load(f)
            index_data = {
                sample["id"]: {k: v for k, v in sample.items() if k != "id"}
                for sample in index_data
            }

    # Load retrieval results
    with open(args.retrieval_results, "r") as f:
        retrieval_results = json.load(f)

    # Get training dataset for generation
    output = [
        get_sample(args, data, index_data, anchor, candidates)
        for anchor, candidates in tqdm(retrieval_results.items())
    ]

    # Save dataset
    with open(args.dataset_output, mode="w") as f:
        json.dump({"version": args.version, "data": output}, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--index_dataset", type=str, required=True)
    parser.add_argument("--retrieval_results", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["none", "random", "sample", "best"], required=True
    )
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--dataset_output", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    args = parser.parse_args()

    build_generate_dataset(args)
