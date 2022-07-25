import pytorch_lightning as pl
from sentence_transformers import losses, SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class Retriever(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = args.model_name
        self.original_model_name = args.original_model_name

        # Initialize original model
        self.model = SentenceTransformer(self.original_model_name)
        self.fc = nn.Linear(1, 1)
        self.tokenizer = self.model.tokenizer

        # Update data module
        data.tokenizer = self.tokenizer
        self.randomize = data.randomize
        self.seed = data.seed

        # Loss
        self.loss = F.mse_loss

    def training_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in [sources, references]
        ]
        output = self.fc(
            torch.cosine_similarity(embeddings[0], embeddings[1]).view(-1, 1)
        )
        loss = self.loss(output, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in [sources, references]
        ]
        output = self.fc(
            torch.cosine_similarity(embeddings[0], embeddings[1]).view(-1, 1)
        )
        loss = self.loss(output, labels)

        self.log("val_loss", loss, prog_bar=True)

        if self.hparams.save_val and not self.hparams.fast_dev_run:
            return {"ids": ids, "labels": labels, "outputs": output, "loss": loss}
        else:
            return loss

    def on_train_epoch_start(self):
        # Print newline to save printed progress per epoch
        print()

        # Randomize dataset on every epoch
        if self.current_epoch > 0:
            self.randomize(self.seed + self.current_epoch)

    def on_validation_epoch_end(self):
        # Print newline to save printed progress after every validation
        print()

    def validation_epoch_end(self, outputs):
        if isinstance(outputs[0], dict):
            outputs = [step["loss"] for step in outputs]

        loss = torch.stack(outputs).mean()

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: Retriever")
        parser.add_argument("--model_name", type=str, default="Retriever")
        parser.add_argument(
            "--original_model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
        )
        parser.add_argument("--from_checkpoint", type=str, default="")
        return parent_parser
