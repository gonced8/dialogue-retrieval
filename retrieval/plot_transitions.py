"""Plot the transition frequency of the dialogue acts
for both, user, agent, and per turn
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from utils import get_sequences

DEBUG = True
PLOT = False
SAVE = True

filename_data = "../data/multiwoz/processed/dev.json"
plot_folder = "../plots/transitions"


def compute_transitions(sequences, return_labels=False):
    # Get correspondance between classes and indices
    classes = set()
    max_length = 0
    for sequence in sequences.values():
        if len(sequence) > max_length:
            max_length = len(sequence)

        for subsequence in sequence:
            classes.update(subsequence)

    class_idx = {
        class_name: i
        for i, class_name in enumerate(["START"] + sorted(classes) + ["END"])
    }
    n_classes = len(class_idx)

    transitions = np.zeros((max_length + 1, n_classes, n_classes), dtype=np.int32)

    for i, sequence in enumerate(sequences.values()):
        # Start transition
        for y in sequence[0]:
            transitions[0, 0, class_idx[y]] += 1

        # Pairs of subsequences
        for j, (previous, current) in enumerate(pairwise(sequence), start=1):
            # Loop previous
            for x in previous:
                # Loop current
                for y in current:
                    transitions[j, class_idx[x], class_idx[y]] += 1

        # End transition
        length = len(sequence)
        for y in sequence[-1]:
            transitions[length, class_idx[y], -1] += 1

    if return_labels:
        return transitions, list(class_idx.keys())
    else:
        return transitions


def plot_transitions(transitions, labels=None, mode="each_turn"):
    if mode == "all_turns":
        transitions = transitions.sum(axis=0)[np.newaxis, ...]
    elif mode == "all_user":
        transitions = transitions[::2, ...].sum(axis=0)[np.newaxis, ...]
    elif mode == "all_agent":
        transitions = transitions[1::2, ...].sum(axis=0)[np.newaxis, ...]

    vmax = transitions.max()
    fig, ax = plt.subplots(figsize=(16, 16))

    for i, matrix in enumerate(transitions):
        img = ax.imshow(matrix, vmin=0, cmap="binary")
        img = ax.imshow(matrix, cmap="cividis")

        if i == 0:
            ax.set_ylabel("before")
            ax.set_xlabel("after")
            if labels is not None:
                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=90)
                ax.set_yticklabels(labels)

                ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        else:
            cb.remove()
        cb = plt.colorbar(img, ax=ax)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.2)

        if mode == "each_turn":
            if i % 2 == 0:
                ax.set_title("User")
            else:
                ax.set_title("Agent")
        else:
            ax.set_title(mode.replace("_", " ").capitalize())

        if PLOT:
            plt.show()
        if SAVE:
            if mode == "each_turn":
                fig.savefig(
                    os.path.join(plot_folder, f"turn_{i}.png"), bbox_inches="tight"
                )
            else:
                fig.savefig(
                    os.path.join(plot_folder, f"{mode}.png"), bbox_inches="tight"
                )


if __name__ == "__main__":
    sequences = get_sequences(filename_data)
    transitions, labels = compute_transitions(sequences, return_labels=True)
    plot_transitions(transitions, labels, mode="all_turns")
    plot_transitions(transitions, labels, mode="all_user")
    plot_transitions(transitions, labels, mode="all_agent")
    plot_transitions(transitions, labels)
