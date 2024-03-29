import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.functional import retrieval_reciprocal_rank

from utils.minmax_ndcg import *


class Retriever(pl.LightningModule):
    def __init__(self, args, data=None):
        super().__init__()
        self.save_hyperparameters(args)

        try:
            self.model_name = self.hparams.model_name
        except AttributeError:
            self.model_name = self.hparams.original_model_name

        # Initialize original model
        self.model = SentenceTransformer(self.hparams.original_model_name)
        self.tokenizer = self.model.tokenizer

        # Update data module
        if data is not None:
            data.tokenizer = self.tokenizer
            self.randomize = data.randomize
            self.seed = data.hparams.seed
        else:
            self.seed = args.seed

        # Loss
        self.loss = F.mse_loss

    def forward(self, x):
        out = self.model(x)["sentence_embedding"]
        # Normalize embeddings because of cosine similarity
        return torch.nn.functional.normalize(out, dim=1)

    def training_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        sources_embeddings, references_embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in [sources, references]
        ]

        # Reshape embeddings
        batch_size = len(ids)
        dialogues_per_sample = references_embeddings.size(0) // batch_size

        if dialogues_per_sample > 1:
            batch_size = sources_embeddings.size(0)
            sources_embeddings = sources_embeddings.unsqueeze(1).expand(
                -1, dialogues_per_sample, -1
            )
            references_embeddings = references_embeddings.view(
                batch_size, dialogues_per_sample, -1
            )

        # Similarity
        output = torch.cosine_similarity(
            sources_embeddings, references_embeddings, dim=-1
        ).unsqueeze(-1)
        output = F.relu(output)

        # Loss
        loss = self.loss(output, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        # Obtain sentence embeddings
        sources_embeddings, references_embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in [sources, references]
        ]

        # Reshape embeddings
        batch_size = len(ids)
        dialogues_per_sample = references_embeddings.size(0) // batch_size

        if dialogues_per_sample > 1:
            batch_size = sources_embeddings.size(0)
            sources_embeddings = sources_embeddings.unsqueeze(1).expand(
                -1, dialogues_per_sample, -1
            )
            references_embeddings = references_embeddings.view(
                batch_size, dialogues_per_sample, -1
            )

        # Similarity
        output = torch.cosine_similarity(
            sources_embeddings, references_embeddings, dim=-1
        )
        output = F.relu(output)

        # Loss
        flatten_output = output.view(-1, 1)
        flatten_labels = labels.view(-1, 1)
        loss = self.loss(flatten_output, flatten_labels)
        self.log("val_loss", loss, batch_size=torch.numel(labels), prog_bar=True)

        mrr = retrieval_reciprocal_rank(
            output, labels.ge(labels.max(-1, keepdim=True)[0])
        )
        score = minmax_ndcg(output, labels)
        metrics = {"mrr": mrr, "minmax_ndcg": score}
        self.log_dict(metrics, prog_bar=True, batch_size=batch_size)

        if self.hparams.save_examples and not self.hparams.fast_dev_run:
            return {"ids": ids, "labels": labels, "outputs": output, "metrics": metrics}
        else:
            return {"metrics": metrics}

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
        metrics = {
            m: torch.stack([step["metrics"][m] for step in outputs]).mean()
            for m in outputs[0]["metrics"]
        }

        self.log_dict(metrics, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: Retriever")
        parser.add_argument(
            "--original_model_name",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
        )
        return parent_parser
