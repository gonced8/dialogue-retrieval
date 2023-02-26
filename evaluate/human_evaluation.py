from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

import pandas as pd


def read_files(files: list[str], data_field: str = None) -> dict:
    """Reads the input JSON files and returns all the data in a dictionary.

    Args:
        files (list[str]): List of input files.
        data_field (str, optional): If the file is a JSON, the data might be in a specific field. Defaults to None.

    Raises:
        ValueError: All the files must be JSON.

    Returns:
        dict: Dictionary where the keys are the filenames (without extension) and the values are lists of dialogues.
    """

    data = {}

    for filename in files:
        filepath = Path(filename)

        # Check if the file is a JSON
        if not filepath.is_file() or filepath.suffix != ".json":
            raise ValueError(f"{filename} must be a JSON file")

        # Read the file
        with open(filename, "r") as f:
            file_data = json.load(f)
            data[filepath.stem] = file_data[data_field] if data_field else file_data

    return data


def format_samples(data: dict, ground_truth: bool = True) -> list[dict]:
    """Merges each sample of all datasets into a single dict, returning a list with all samples.

    Args:
        data (dict): Dictionary where the keys are the filenames (without extension) and the values are lists of dialogues.
        ground_truth (bool, optional): Wheather to include the ground truth resposne in the answers. Defaults to True.

    Raises:
        ValueError: If the datasets are misaligned (might have different samples).

    Returns:
        list[dict]: List where each sample consists of a id, context and responses (ground truth and models)
    """

    dataset = []
    names = data.keys()

    for samples in zip(*data.values()):
        if len(set(sample["id"] for sample in samples)) > 1:
            raise ValueError("Mismatch between samples ids.")

        dataset.append(
            {
                "id": samples[0]["id"],
                "context": samples[0]["context"],
                "response": {"ground_truth": samples[0]["response"]}
                if ground_truth
                else {}
                | {
                    name: sample["model_answer"] for name, sample in zip(names, samples)
                },
            }
        )

    return dataset


def shuffle_responses():
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input", nargs="+", type=str, help="Files with text to be evaluated."
    )
    parser.add_argument(
        "--ground_truth",
        default=False,
        action=BooleanOptionalAction,
        help="Include ground truth responses.",
    )
    parser.add_argument(
        "--data_field",
        type=str,
        default=None,
        help="If the input file is a JSON, the data might be in a particular field.",
    )
    parser.add_argument(
        "-seed", type=str, default=42, help="Seed to initialize randomness."
    )
    args = parser.parse_args()

    data = read_files(args.input, args.data_field)
    data = format_samples(data, args.ground_truth)
    data = shuffle_responses(data, args.seed)

    print(json.dumps(data, indent=4))
