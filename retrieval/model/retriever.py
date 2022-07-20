import json
import os

import pytorch_lightning as pl
from sentence_transformers import losses, SentenceTransformer
import torch
import torch.nn as nn
from torch.optim import AdamW


class Retriever(pl.LightningModule):
    def __init__(self, args, data):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = args.model_name
        self.original_model_name = args.original_model_name

        # Initialize original model
        self.model = SentenceTransformer(self.original_model_name)
        self.tokenizer = self.model.tokenizer

        # Update data module
        data.tokenizer = self.tokenizer
        self.randomize = data.randomize
        self.seed = data.seed

        # Loss
        self.loss = losses.CosineSimilarityLoss(
            model=self.model, cos_score_transformation=lambda x: (x + 1) * 0.5
        )

    def training_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        loss = self.loss([sources, references], labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ids, sources, references, labels = batch.values()

        if self.hparams.save_val:
            embeddings = [
                self.loss.model(sentence_feature)["sentence_embedding"]
                for sentence_feature in [sources, references]
            ]
            output = self.loss.cos_score_transformation(
                torch.cosine_similarity(embeddings[0], embeddings[1])
            )
            loss = self.loss.loss_fct(output, labels.view(-1))
        else:
            loss = self.loss([sources, references], labels)

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

    def validation_epoch_end(self, outputs):
        if self.hparams.save_val and not self.hparams.fast_dev_run:
            loss = torch.stack([step["loss"] for step in outputs]).mean()
            data = [{"val_loss": loss.item()}]
            data.extend(
                [
                    {"id": sample_id, "label": label.item(), "output": output.item()}
                    for step in outputs
                    for sample_id, label, output in zip(
                        step["ids"],
                        step["labels"],
                        step["outputs"],
                    )
                ]
            )

            # Round float numbers
            data = [
                {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in sample.items()
                }
                for sample in data
            ]

            # Save results to file
            output_filename = os.path.join(
                self.trainer.logger.log_dir, "results_val.json"
            )

            with open(output_filename, "w") as f:
                json.dump(data, f, indent=4)
        else:
            loss = torch.stack(outputs)

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
