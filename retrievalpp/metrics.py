import json

import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from tqdm.contrib import tzip


class RetrievalMetrics:
    def __init__(self, index_dataset, query_dataset, output=None):
        # Load datasets
        with open(index_dataset, "r") as f:
            self.index_dataset = json.load(f)["data"]

        with open(query_dataset, "r") as f:
            self.query_dataset = json.load(f)["data"]

        # Name of file where output metrics and retrieval results will be saved
        self.output = output

        # Load metrics
        self.bleu = BLEU(effective_order=True)
        self.rouge = RougeScorer(["rougeL"])

    def __call__(self, eval_pred):
        candidates_ids = eval_pred.predictions
        candidates = [
            [self.index_dataset[idx]["text"][-1] for idx in sample_candidates]
            for sample_candidates in candidates_ids
        ]

        answers_ids = eval_pred.label_ids
        answers = [self.query_dataset[idx]["text"][-1] for idx in answers_ids]

        # Compute BLEUo scores
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

        # Get scores from best candidate and average
        best_bleu = np.argmax(bleu_scores, axis=1)
        best_rouge = np.argmax(bleu_scores, axis=1)

        bleu_score = bleu_scores[best_bleu].mean()
        rouge_score = rouge_scores[best_rouge].mean()

        # Compute MRR from BLEU and ROUGE
        mrr_bleu = np.reciprocal(best_bleu + 1.0).mean()
        mrr_rouge = np.reciprocal(best_rouge + 1.0).mean()

        metrics = {
            "bleu": bleu_score,
            "rouge": rouge_score,
            "mrr_bleu": mrr_bleu,
            "mrr_rouge": mrr_rouge,
        }

        if self.output is not None:
            # Retrieval results
            results = [
                {
                    "id": self.query_dataset[query_id]["id"],
                    "context": self.query_dataset[query_id]["text"][:-1],
                    "knowledge": sample_candidates,
                    "response": self.query_dataset[query_id]["text"][-1],
                }
                for query_id, sample_candidates in zip(answers_ids, candidates)
            ]

            # Save to file
            with open(self.output, "w") as f:
                json.dump({"version": self.output, "metrics": metrics, "data": results})

        return metrics
