import json
import matplotlib.pyplot as plt
import numpy as np
import os

dir_pairs = "../../data/multiwoz/processed/"
dir_hists = "../../plots/"
SHOW = False
SAVE = True

if __name__ == "__main__":
    for root, _, files in os.walk(dir_pairs):
        for name in files:
            if not name.startswith("pairs_") or not name.endswith(".json"):
                continue

            filename = os.path.join(root, name)
            annotations = os.path.splitext(name)[0].split("_")[-1].split("+")

            with open(filename, "r") as f:
                pairs = json.load(f)

            fig, ax = plt.subplots()

            ax.hist(
                pairs.values(),
                bins=np.linspace(0, 1, 20),
                weights=np.ones(len(pairs)) / len(pairs),
            )
            ax.set_xlabel("Sequences Levenshtein Distance")
            ax.set_ylabel("Relative Frequency")
            ax.set_title(" + ".join([a.capitalize() for a in annotations]))

            # Show and/or save plots
            plt.tight_layout()
            if SAVE:
                name = os.path.splitext(name)[0]
                filename_hist = os.path.join(dir_hists, f"hist_{name}.pdf")
                plt.savefig(filename_hist, bbox_inches="tight")
                print(f"Saved: {filename_hist}")

    if SHOW:
        plt.show()
