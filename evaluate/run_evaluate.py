from argparse import ArgumentParser, BooleanOptionalAction
import json

from datasets import load_dataset
import evaluate
import numpy as np
from sacrebleu import corpus_bleu

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--delexicalized", default=False, action=BooleanOptionalAction)
    parser.add_argument("--retrieved", default=False, action=BooleanOptionalAction)
    args = parser.parse_args()

    # Load results
    data_files = {"test": args.results}
    results = load_dataset("json", data_files=data_files, field="data")

    references = [
        sample["delexicalized"] if args.delexicalized else sample["truth_answer"]
        for sample in results["test"]
    ]
    predictions = [
        sample["knowledge"][0] if args.retrieved else sample["model_answer"]
        for sample in results["test"]
    ]

    # Evaluate
    print("Computing BLEU...")
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)

    # print("Computing SacreBLEU...")
    # sacrebleu_score = corpus_bleu(predictions, [references]).score

    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    # print("Computing BERTScore...")
    # bertscore = evaluate.load("bertscore")
    # bertscore_score = bertscore.compute(
    #    predictions=predictions,
    #    references=references,
    #    lang="en",
    #    rescale_with_baseline=True,
    #    batch_size=args.batch_size,
    #    use_fast_tokenizer=True,
    # )

    # print("Computing BLEURT...")
    # bleurt = evaluate.load("bleurt", module_type="metric", checkpoint="BLEURT-20")
    # bleurt_score = bleurt.compute(
    #    predictions=predictions,
    #    references=references,
    # )

    # Average
    # bertscore_score = {
    #    k: np.mean(v).item() for k, v in bertscore_score.items() if k != "hashcode"
    # }
    # bleurt_score = {k: np.mean(v).item() for k, v in bleurt_score.items()}

    # Save scores
    scores = {
        "bleu": bleu_score,
        # "sacrebleu": sacrebleu_score,
        "rouge": rouge_score,
        # "bertscore": bertscore_score,
        # "bleurt": bleurt_score,
    }
    print(json.dumps(scores, indent=4))

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=4)
