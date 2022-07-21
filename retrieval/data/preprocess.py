import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from transformers import AutoTokenizer, AutoModel

from .multiwoz import *

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

    labels = np.array([sample["similarity"] for sample in dataset]).reshape(-1, 1)

    qt = QuantileTransformer(n_quantiles=10, random_state=0)
    labels_transformed = qt.fit_transform(labels)

    fig, ax = plt.subplots()
    ax.hist(labels, bins=20, alpha=0.5, label="original")
    ax.hist(labels_transformed, bins=20, label="transformed")
    fig.tight_layout()
    plt.show()
