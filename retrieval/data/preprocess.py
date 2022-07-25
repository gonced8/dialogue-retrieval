import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from transformers import AutoTokenizer, AutoModel

from .multiwoz import *

SAVE_MODEL = True
SHOW = False
SAVE = True

model_folder = "data"
plot_folder = "../plots"

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = MultiWOZ.add_argparse_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="seed")
    args = parser.parse_args()

    data = MultiWOZ(args)
    data.prepare_data()
    data.setup("train")
    dataset = data.train_dataset

    labels = np.array(
        [sample["similarity"] for sample in tqdm(dataset, desc="Getting labels")]
    ).reshape(-1, 1)

    # Fit model
    qt = QuantileTransformer(n_quantiles=10, random_state=42)
    labels_transformed = qt.fit_transform(labels)

    if SAVE_MODEL:
        joblib.dump(qt, os.path.join(model_folder, "quantile_transformer.joblib"))

    # Plot
    if SHOW or SAVE:
        fig, ax = plt.subplots()

        ax.hist(labels, bins=20, alpha=0.5, label="original")
        ax.hist(labels_transformed, bins=20, alpha=0.5, label="transformed")

        ax.set_xlabel("Similarity")
        ax.set_ylabel("Frequency")
        ax.legend()
        fig.tight_layout()

    if SAVE:
        fig.savefig(
            os.path.join(plot_folder, f"quantile_transformer.pdf"), bbox_inches="tight"
        )

    if SHOW:
        plt.show()
