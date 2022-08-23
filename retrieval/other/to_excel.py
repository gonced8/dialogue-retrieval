from argparse import ArgumentParser
import json
import pandas as pd


def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    # Ignore initial stats
    return data[1:]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=str, nargs="+", required=True, help="JSON file with results"
    )
    parser.add_argument("--labels", type=str, nargs="*", help="Labels to use in plot")
    parser.add_argument(
        "--output",
        type=str,
        help="Output Excel file",
    )
    args = parser.parse_args()

    data = {}
    for filename, label in zip(args.input, args.labels if args.labels else args.input):
        data[label] = pd.DataFrame(load_data(filename))

    with pd.ExcelWriter("output.xlsx") as writer:
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name)
