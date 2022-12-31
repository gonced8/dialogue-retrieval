from argparse import ArgumentParser
from functools import partial
from itertools import islice
import json

from rouge_score import rouge_scorer
from tqdm import tqdm

from utils.annotations import compare_lcs, compare_lcspp


def compute_similarity(anchor, candidate, st, rouge):
    text_similarity = {
        "st": float(st),
        "answer_rougeL": rouge.score(
            anchor["text"].rsplit("\n", 1)[1],
            candidate["text"].rsplit("\n", 1)[1],
        )["rougeL"].fmeasure,
        "rougeL": rouge.score(
            anchor["text"],
            candidate["text"],
        )["rougeL"].fmeasure,
    }

    if "annotations" in anchor:
        annotations_similarity = {
            "lcs": compare_lcs(
                anchor["annotations"],
                candidate["annotations"],
            ),
            "lcs++": compare_lcspp(
                anchor["annotations"],
                candidate["annotations"],
            ),
        }

        text_similarity.update(annotations_similarity)

    return text_similarity


def compute_train_dataset(args):
    # Load train dataset
    with open(args.train_data, "r") as f:
        train_data = json.load(f)
        train_data = {
            sample["id"]: {k: v for k, v in sample.items() if k != "id"}
            for sample in train_data
        }

    # Load results of similar samples from pretrained Sentence Transformer
    with open(args.results_st, "r") as f:
        train_dataset = json.load(f)

    # Load ROUGE metric
    rouge = rouge_scorer.RougeScorer(["rougeL"])

    # Compute similarity between anchor and samples
    train_dataset = {
        anchor: {
            candidate: compute_similarity(
                train_data[anchor], train_data[candidate], st, rouge
            )
            for candidate, st in islice(candidates.items(), 0, args.top_k, 1)
        }
        for anchor, candidates in tqdm(
            train_dataset.items(),
            desc="Computing similarities for each training sample",
        )
    }

    # Round floats to 4 digits and save
    train_dataset = {
        anchor: {
            candidate: {metric: round(value, 4) for metric, value in metrics.items()}
            for candidate, metrics in candidates.items()
        }
        for anchor, candidates in train_dataset.items()
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
