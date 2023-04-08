from datetime import datetime
import json
from pathlib import Path

import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from tqdm.contrib import tzip


class RetrievalMetrics:
    def __init__(
        self,
        index_dataset,
        query_dataset,
        field="data",
        output=None,
        delexicalized=False,
    ):
        self.delexicalized = delexicalized

        # Load datasets
        with open(index_dataset, "r") as f:
            self.index_dataset = json.load(f)
            if field:
                self.index_dataset = self.index_dataset[field]

        with open(query_dataset, "r") as f:
            self.query_dataset = json.load(f)
            if field:
                self.query_dataset = self.query_dataset[field]

        # Name of folder where output metrics and retrieval results will be saved
        self.output = output

        # Load metrics
        self.bleu = BLEU(effective_order=True)
        self.rouge = RougeScorer(["rougeL"])

    def __call__(self, eval_pred):
        candidates_ids, queries_ids = eval_pred

        # Gather dataset text
        dataset = [
            {
                "id": self.query_dataset[query_id]["id"],
                "context": self.query_dataset[query_id]["context"],
                "response": self.query_dataset[query_id]["response"],
                "delexicalized": self.query_dataset[query_id]["delexicalized"]
                if "delexicalized" in self.index_dataset[0]
                else None,
                "knowledge": [
                    self.index_dataset[idx]["delexicalized"]
                    if self.delexicalized and "delexicalized" in self.index_dataset[0]
                    else self.index_dataset[idx]["response"]
                    for idx in sample_candidates
                ],
            }
            for query_id, sample_candidates in zip(queries_ids, candidates_ids)
        ]

        # Strip system initial token
        candidates = [
            [candidate.lstrip("System: ") for candidate in sample["knowledge"]]
            for sample in dataset
        ]
        answers = [
            (
                sample["delexicalized"]
                if self.delexicalized and "delexicalized" in self.index_dataset[0]
                else sample["response"]
            ).lstrip("System: ")
            for sample in dataset
        ]

        # Compute BLEU scores
        bleu_scores = np.array(
            [
                [
                    self.bleu.sentence_score(candidate, [answer]).score
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
                    self.rouge.score(answer, candidate)["rougeL"].fmeasure
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

        # Save validation results
        if self.output:
            if ".json" in self.output:
                output_file = self.output
            else:
                # Create folder if it doesn't exist
                output_folder = Path(self.output)
                output_folder.mkdir(parents=True, exist_ok=True)

                output_file = output_folder / (
                    datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".json"
                )

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "version": output_folder.name,
                        "metrics": metrics,
                        "data": dataset,
                    },
                    f,
                    indent=4,
                    ensure_ascii=False,
                )
            print(f"Saved to: {output_file}")

        return metrics
