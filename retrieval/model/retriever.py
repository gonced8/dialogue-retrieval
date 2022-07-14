import pytorch_lightning as pl
from sentence_transformers import losses, SentenceTransformer
import torch.nn as nn
from transformers.optimization import AdamW


class Retriever(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = "Retriever"
        self.original_model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # Initialize original model
        self.model = SentenceTransformer(self.original_model_name)
        self.tokenizer = self.model.tokenizer

        # Update data module
        data.tokenizer = self.tokenizer
        self.randomize = data.randomize
        self.seed = data.seed

        # Loss
        self.loss = losses.CosineSimilarityLoss(model=self.model)

    def training_step(self, batch, batch_idx):
        sources, references, labels = batch.values()

        labels = 2 * labels - 1
        loss = self.loss([sources, references], labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sources, references, labels = batch.values()

        labels = 2 * labels - 1
        loss = self.loss([sources, references], labels)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Print newline to save printed progress per epoch
        print()

        # Randomize dataset on every epoch
        if self.current_epoch > 0:
            self.randomize(self.seed + self.current_epoch)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: Retriever")
        parser.add_argument("--model_name", type=str, default="retriever")
        parser.add_argument("--from_checkpoint", type=str, default="")
        return parent_parser
