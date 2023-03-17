from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_from_xlsx(
    input_filepath: str, order_filepath: str = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the human evaluation from the .xlsx file.
    If the order file is not specified, use the default path.

    Args:
        input_filepath (str): Path of the .xlsx file with the human evaluation.
        order_filepath (str, optional): Path of the .xlsx file with the orders. Defaults to input_order.xlsx.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Returns the human evaluation and the order of each column.
    """
    data = pd.read_excel(input_filepath)

    if order_filepath is None:
        filepath = Path(input_filepath)
        order_filepath = filepath.with_stem(filepath.stem + "_order")

    order = pd.read_excel(order_filepath, header=None)

    return data, order


def get_scores(data: pd.DataFrame, order: pd.DataFrame) -> pd.DataFrame:
    n_models = len(order.columns)

    # Select only relevant rows and columns
    data = data.iloc[:, -n_models:]
    data.dropna(how="all", inplace=True)

    # Get models names
    names = sorted(order.iloc[0].tolist())

    # Create DataFrame for scores
    scores = pd.DataFrame(np.zeros(data.shape), columns=names, dtype=np.int32)

    # Get scores
    for idx, ranks in data.iterrows():
        models = order.loc[idx]
        n_pos = ranks.notna().sum()
        n_pos = n_pos - 1 if n_pos > 1 else 1

        for pos, model_numbers in enumerate(ranks.astype(str)):
            if model_numbers == "nan":
                continue

            model_names = [
                models[int(float(number)) - 1] for number in model_numbers.split()
            ]
            scores.iloc[idx][model_names] = 1 - pos / n_pos

    return scores


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="File (.xlsx) with human annotations.",
    )
    parser.add_argument(
        "--input_order",
        type=str,
        default=None,
        help="File (.xlsx) with order of columns in input file (because it is shuffled).",
    )
    parser.add_argument(
        "--output", type=str, help="Filepath to save the results (.json)."
    )
    args = parser.parse_args()

    # Load data
    data, order = load_from_xlsx(args.input, args.input_order)

    # Get scores
    scores = get_scores(data, order)

    # Compute average
    print(scores.mean(axis=0))
