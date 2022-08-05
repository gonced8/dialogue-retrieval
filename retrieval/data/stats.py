import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModel

from .lcs import *
from .multiwoz import *

SAVE = False
SHOW = True


def scatter_hist(x, y, xlabel=None, ylabel=None):
    # start with a square Figure
    fig = plt.figure(figsize=(8, 8))

    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(7, 2),
        height_ratios=(2, 7),
        left=0.1,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.05,
        hspace=0.05,
    )

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=10, alpha=0.2)

    # now determine nice limits by hand:
    binwidth = 0.02
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = np.ceil(xymax / binwidth) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation="horizontal")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    return fig


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZ.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()

    data = MultiWOZ(args)
    data.prepare_data()
    data.setup("fit")
    dataset = data.val_dataset

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    dlengths1 = []
    dlengths2 = []
    dlengths3 = []
    similarities = []

    for sample in tqdm(dataset, desc="Iterating samples"):
        d_ids = sample["d_ids"]
        dialogues = sample["dialogues"]
        annotations = ["domains", "acts", "slots", "values"]

        # Get conversations, tokenize, and calculate length difference
        conversations = [MultiWOZDataset.get_conversation(d) for d in dialogues]
        l1, l2 = [len(tokenizer.tokenize(c)) for c in conversations]
        dlength1 = abs(l1 - l2) / (l1 + l2)

        # Get sequences, calculate length difference, and compute similarity
        sequences = [
            MultiWOZDataset.get_sequence(d, annotations, flatten=True)
            for d in dialogues
        ]
        l3, l4 = [len(s) for s in sequences]
        dlength2 = abs(l3 - l4) / (l3 + l4)
        similarity = lcs_similarity(*sequences)

        # Calculate difference of number of turns
        l5, l6 = [len(d) for d in dialogues]
        dlength3 = abs(l5 - l6) / (l5 + l6)

        # Append values to lists
        dlengths1.append(dlength1)
        dlengths2.append(dlength2)
        dlengths3.append(dlength3)
        similarities.append(similarity)

    # Plot similarity vs length differences
    fig1 = scatter_hist(
        dlengths1,
        similarities,
        "Normalized text length difference",
        "LCS similarity",
    )
    fig2 = scatter_hist(
        dlengths2,
        similarities,
        "Normalized number of annotations difference",
        "LCS similarity",
    )
    fig3 = scatter_hist(
        dlengths3,
        similarities,
        "Normalized number of turns difference",
        "LCS similarity",
    )

    if SAVE:
        fig1.savefig(
            "../plots/similarity_text_length_difference.png", bbox_inches="tight"
        )
        fig2.savefig(
            "../plots/similarity_number_annotations_difference.png",
            bbox_inches="tight",
        )
        fig3.savefig(
            "../plots/similarity_number_turns_difference.png", bbox_inches="tight"
        )

    if SHOW:
        plt.show()
