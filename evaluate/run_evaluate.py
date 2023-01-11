from argparse import ArgumentParser
import json

from datasets import load_dataset
import evaluate
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    # Load results
    data_files = {"test": args.results}
    results = load_dataset("json", data_files=data_files, field="data")

    references = [sample["truth_answer"] for sample in results["test"]]
    predictions = [sample["model_answer"] for sample in results["test"]]

    # Evaluate
    print("Computing BLEU...")
    bleu_score = bleu.compute(predictions=predictions, references=references)
    print("Computing ROUGE...")
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )
    print("Computing BERTScore...")
    bertscore_score = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        rescale_with_baseline=True,
        batch_size=args.batch_size,
        use_fast_tokenizer=True,
    )

    # Average
    bertscore_score = {
        k: np.mean(v).item() for k, v in bertscore_score.items() if k != "hashcode"
    }

    # Save scores
    scores = {"bleu": bleu_score, "rouge": rouge_score, "bertscore": bertscore_score}
    print(json.dumps(scores, indent=4))

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=4)
