from argparse import ArgumentParser
import json

from tqdm import tqdm

from utils import slice_dict
from utils.lcs import lcs_similarity


def similarity_fn(x, y):
    # Flatten
    x = [label for turn in x for label in turn]
    y = [label for turn in y for label in turn]

    return lcs_similarity(x, y)


def compute_train_dataset(args):
    # Load train dataset
    with open(args.train_data, "r") as f:
        train_data = json.load(f)
        train_data = {
            sample["id"]: {
                "text": sample["text"],
                "annotations": sample["annotations"],
            }
            for sample in train_data
        }

    # Load results of similar samples from pretrained Sentence Transformer
    with open(args.results_st, "r") as f:
        train_dataset = json.load(f)

    # Compute LCS between anchor and samples
    similarity = [
        {
            candidate: similarity_fn(
                train_data[anchor]["annotations"],
                train_data[candidate]["annotations"],
            )
            for candidate in slice_dict(candidates, stop=args.top_k)
        }
        for anchor, candidates in tqdm(
            train_dataset.items(), desc="Computing annotation similarity"
        )
    ]

    train_dataset = {
        anchor: {
            candidate: {"st": st, "lcs": lcs}
            for (candidate, st), lcs in zip(candidates.items(), lcs_scores.values())
        }
        for (anchor, candidates), lcs_scores in zip(train_dataset.items(), similarity)
    }

    with open(args.dataset_output, "w") as f:
        json.dump(train_dataset, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--results_st", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--dataset_output", type=str, required=True)
    args = parser.parse_args()

    compute_train_dataset(args)
