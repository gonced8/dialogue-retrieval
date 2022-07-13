import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from transformers.optimization import AdamW

import torch.nn as nn


class Retriever(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = "Retriever"
        self.original_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Initialize original model
        self.model = SentenceTransformer(self.original_model_name)

        # Loss
        self.loss = nn.MSELoss(reduction="mean")

    def training_step(self, batch, batch_idx):
        print(batch)
        input()

    # TODO randomize dataset with seed+epoch in the end of each epoch

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: Retriever")
        parser.add_argument("--model_name", type=str, default="retriever")
        parser.add_argument("--from_checkpoint", type=str, default="")
        return parent_parser
