"""Plot how some metrics (i.e., ROUGE, LCS score) vary with
the number of turns of the original dialogue and retrieved dialogue
"""

from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns


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
        help="Beginning of saved plots filenames (e.g., plots/output_plot)",
    )
    parser.add_argument(
        "command",
        nargs="+",
        choices=["save", "show"],
        help="Save and/or show plots",
    )
    args = parser.parse_args()

    # Load data files
    data = [pd.DataFrame(load_data(filename)) for filename in args.input]
    for df, label in zip(data, args.labels if args.labels else args.input):
        df["label"] = label

    # Concatenate dataframes
    df = pd.concat(data, axis=0, ignore_index=True)

    # Compute number of turns
    df["nturns_truth"] = df["truth"].map(lambda x: len(x.split("\n")))
    df["nturns_retrieved"] = df["retrieved"].map(lambda x: len(x.split("\n")))
    df["nturns_diff"] = df["nturns_truth"] - df["nturns_retrieved"]

    # Plot RougeL-P in function of # turns of truth dialogue
    g1 = sns.relplot(
        data=df,
        x="nturns_truth",
        y="answer_rougeL",
        hue="label",
        kind="line",
        aspect=1.33,
    )
    g1.set_axis_labels("# turns", "RougeL-P")
    for ax in g1.axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    # Plot RougeL-P in function of difference of # turns between truth and retrieved
    g2 = sns.relplot(
        data=df,
        x="nturns_diff",
        y="answer_rougeL",
        hue="label",
        kind="line",
        aspect=1.33,
    )
    g2.set_axis_labels("# turns truth - # turns retrieved", "RougeL-P")
    for ax in g2.axes.flat:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    if "save" in args.command:
        g1.fig.savefig(args.output + "_truth.pdf", bbox_inches="tight")
        g2.fig.savefig(args.output + "_diff.pdf", bbox_inches="tight")
