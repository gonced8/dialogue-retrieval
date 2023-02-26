from argparse import ArgumentParser
import json

from datasets import load_dataset
import numpy as np
from pyserini.search.lucene import LuceneSearcher
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from tqdm.contrib import tzip


def build_samples(samples):
    return {
        "id": samples["id"],
        "contents": [" ".join(turns) for turns in samples["context"]],
    }


if __name__ == "__main__":
    # Arguments
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["data", "test"])
    parser.add_argument("--index_dataset", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--preprocess_batch_size", type=int, default=256)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_candidates", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "data":
        # Load dataset
        dataset = load_dataset("json", data_files=args.index_dataset, field="data")[
            "train"
        ]

        # Prepare documents to be index
        dataset = dataset.map(
            build_samples,
            batched=True,
            batch_size=args.preprocess_batch_size,
            remove_columns=dataset.column_names,
        )

        # Save documents
        with open(args.output, "w") as f:
            json.dump([sample for sample in dataset], f, indent=4)

    if args.mode == "test":
        # Load dataset
        with open(args.input, "r") as f:
            dataset = json.load(f)["data"]

        # Load index
        searcher = LuceneSearcher(args.index)

        with open(args.index_dataset, "r") as f:
            index_dataset = json.load(f)["data"]

        index_dataset = {
            sample["id"]: {k: v for k, v in sample.items() if k != "id"}
            for sample in index_dataset
        }

        # Retrieve
        for sample in tqdm(dataset, desc="Retrieving using BM25"):
            # TODO: exclude indices of same dialogue
            hits = searcher.search(q=" ".join(sample["context"]), k=args.n_candidates)
            sample["knowledge"] = [
                index_dataset[hit.docid]["delexicalized"] for hit in hits
            ]

        # Load metrics
        bleu = BLEU(effective_order=True)
        rouge = RougeScorer(["rougeL"])

        # Strip system initial token
        candidates = [
            [candidate.lstrip("System: ") for candidate in sample["knowledge"]]
            for sample in dataset
        ]
        answers = [sample["delexicalized"].lstrip("System: ") for sample in dataset]

        # Compute BLEU scores
        bleu_scores = np.array(
            [
                [
                    bleu.sentence_score(candidate, [answer]).score
                    for candidate in sample_candidates
                ]
                for sample_candidates, answer in tzip(
                    candidates, answers, desc="Computing BLEU"
                )
            ]
        )

        # Compute ROUGE scores
        rouge_scores = np.array(
            [
                [
                    rouge.score(answer, candidate)["rougeL"].fmeasure
                    for candidate in sample_candidates
                ]
                for sample_candidates, answer in tzip(
                    candidates, answers, desc="Computing ROUGE"
                )
            ]
        )

        # Get average scores for all candidates
        bleu_score = bleu_scores.mean()
        rouge_score = rouge_scores.mean()

        # Get average scores for best candidates
        bleu_best = np.argmax(bleu_scores, axis=1)
        rouge_best = np.argmax(rouge_scores, axis=1)

        bleu_best_score = bleu_scores[:, bleu_best].mean()
        rouge_best_score = rouge_scores[:, rouge_best].mean()

        # Compute MRR from BLEU and ROUGE
        mrr_bleu = np.reciprocal(bleu_best + 1.0).mean()
        mrr_rouge = np.reciprocal(rouge_best + 1.0).mean()

        metrics = {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "bleu_best": bleu_best_score,
            "rouge_best": rouge_best_score,
            "mrr_bleu": mrr_bleu,
            "mrr_rouge": mrr_rouge,
        }

        print(metrics)

        # Save results to file
        if args.output is not None:
            with open(args.output, "w") as f:
                json.dump(
                    {"metrics": metrics, "data": [sample for sample in dataset]},
                    f,
                    indent=4,
                )

            print(f"Saved results to: {args.output}")
