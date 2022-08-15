import json
from pathlib import Path

import faiss
import faiss.contrib.torch_utils
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class NextTurn(pl.LightningModule):
    def __init__(self, args, data=None):
        super().__init__()
        self.save_hyperparameters(args)

        try:
            self.model_name = self.hparams.model_name
        except AttributeError:
            self.model_name = self.hparams.original_model_name

        # Load encoder
        self.model = SentenceTransformer(self.hparams.original_model_name)
        self.tokenizer = self.model.tokenizer

        # Load index
        index_directory = Path(self.hparams.index_directory)
        self.index = faiss.read_index(str(index_directory / "index.bin"))
        with open(index_directory / "ids_labels.json", "r") as f:
            self.ids_labels = list(json.load(f).values())
        self.data = data

        # Update data module
        if data is not None:
            data.tokenizer = self.tokenizer
            self.randomize = data.randomize
            self.seed = data.hparams.seed
        else:
            self.seed = args.seed

    def forward(self, x):
        # Compute embeddings and normalize (because of cosine simlarity)
        out = self.model(x)["sentence_embedding"]
        return torch.nn.functional.normalize(out, dim=1)

    def test_step(self, batch, batch_idx):
        # Compute dialogue embeddings
        embeddings = self.forward(batch["contexts_tokenized"])

        print(embeddings)

        # Retrieve similar dialogues
        distances, indices = self.index.search(embeddings.cpu(), self.hparams.k)
        candidates = [
            {self.ids_labels[idx]: distance for idx, distance in zip(*sample)}
            for sample in zip(indices, distances)
        ]
        answers = [
            self.data.test_dataset.from_id(next(iter(candidates_i)))[-1]
            for candidates_i in candidates
        ]

        print(answers)
        input()

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: NextTurn")
        parser.add_argument(
            "--original_model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
        )
        parser.add_argument(
            "--index_directory", type=str, default="data/multiwoz/index"
        )
        parser.add_argument("--k", type=int, default=10)
        return parent_parser
