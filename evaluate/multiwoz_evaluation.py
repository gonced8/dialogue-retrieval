from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
import json

from datasets import load_dataset
from mwzeval.metrics import Evaluator
from tqdm import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("--retrieved", default=False, action=BooleanOptionalAction)
    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    # Load results
    data_files = {"test": args.input}
    data = load_dataset("json", data_files=data_files, field="data")

    # Gather predictions in correct format
    my_predictions = defaultdict(list)

    for sample in tqdm(data["test"], desc="Gathering results"):
        dialog_id = sample["id"].rsplit("_", 1)[0].lower().rstrip(".json")
        response = (
            sample["knowledge"][0] if args.retrieved else sample["model_answer"]
        ).split(": ", 1)[1]
        my_predictions[dialog_id].append({"response": response})

    # Evaluate
    e = Evaluator(bleu=True, success=False, richness=False)
    results = e.evaluate(my_predictions)
    print(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
