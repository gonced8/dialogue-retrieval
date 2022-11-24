import json
from pathlib import Path

from datasets import load_metric
from faiss import read_index
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchmetrics.functional import retrieval_reciprocal_rank

from utils import parse_rouge_score
from utils.minmax_ndcg import *


class RetrieverAnswererer(pl.LightningModule):
    def __init__(self, args, data=None):
        super().__init__()
        self.save_hyperparameters(args)

        self.model_name = self.hparams.model_name

        # Initialize retriever encoder
        self.retriever_encoder = SentenceTransformer(self.hparams.retriever_encoder)
        self.retriever_tokenizer = self.retriever_encoder.tokenizer

        # Load index
        if self.hparams.index_directory is not None:
            self.index = read_index(
                str(Path(self.hparams.index_directory) / "index.bin")
            )
            with open(Path(self.hparams.index_directory) / "ids_labels.json", "r") as f:
                self.ids_labels = list(json.load(f).values())
            with open(self.hparams.index_dataset, "r") as f:
                self.index_dataset = json.load(f)

        # Update data module
        if data is not None:
            data.tokenizer = self.retriever_tokenizer

        self.seed = self.hparams.seed

        # Loss
        self.triplet_loss = torch.nn.TripletMarginLoss()

        # Metrics
        self.rouge_metric = load_metric("rouge")

    def retrieve(self, batch):
        # Encode
        embeddings = self.retriever_encoder(batch["context_tokenized"])[
            "sentence_embedding"
        ]

        # Normalize embeddings because of cosine similarity
        # embeddings = torch.nn.functional.normalize(embeddings, dim=1)

        # Retrieve
        distances, indices = self.index.search(
            embeddings.cpu().detach().numpy(),
            self.hparams.n_candidates + batch.get("max_n_subdialogue", 0),
        )

        # Filter sub-dialogues hits from the same original dialogue
        query_base_ids = [query_id.split("_")[0] for query_id in batch["ids"]]
        candidates = [
            [
                {
                    "id": hit_id,
                    "score": hit_distance,
                    "dialogue": self.index_dataset[hit_id],
                }
                for hit_id, hit_distance in zip(
                    hits_ids,
                    hits_distances,
                )
                if not self.ids_labels[hit_id].startswith(query_base_id)
            ]
            for hits_ids, hits_distances, query_base_id in zip(
                indices,
                distances,
                query_base_ids,
            )
        ]

        # Limit to a total of n_candidates per sample
        candidates = [hits[: self.hparams.n_candidates] for hits in candidates]

        return candidates

    def forward(self, batch):
        candidates = self.retrieve(batch)
        return candidates

    def on_train_epoch_start(self):
        # Print newline to save printed progress per epoch
        print()

    def training_step(self, batch, batch_idx):
        anchor = self.self.retriever_encoder(batch["anchor"])["sentence_embedding"]
        positive = self.self.retriever_encoder(batch["positive"])["sentence_embedding"]
        negative = self.self.retriever_encoder(batch["negative"])["sentence_embedding"]

        output = self.triplet_loss(anchor, positive, negative)
        return output

    # def validation_step(self, batch, batch_idx):
    #    raise NotImplementedError

    def validation_epoch_end(self, outputs):
        raise NotImplementedError

    def on_validation_epoch_end(self):
        # Print newline to save printed progress after every validation
        print()

    def test_step(self, batch, batch_idx):
        candidates = self.forward(batch)
        texts = [
            sample_candidates[0]["dialogue"]["text"] for sample_candidates in candidates
        ]

        # Get truth and model answers and remove "SYSTEM: "
        start = len("SYSTEM: ")
        truth_answers = [answer[start:] for answer in batch["answers"]]
        model_answers = [text.rsplit("\n", 1)[1][start:] for text in texts]

        # Compute metrics
        self.rouge_metric.add_batch(predictions=model_answers, references=truth_answers)

        return

    def test_epoch_end(self, outputs):
        rouge_score = self.rouge_metric.compute()
        rouge_score = parse_rouge_score(rouge_score)

        self.log_dict(rouge_score, prog_bar=True)
        return

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("Module: RetrieverAnswerer")
        parser.add_argument(
            "--retriever_encoder",
            type=str,
            default="sentence-transformers/all-mpnet-base-v2",
        )
        parser.add_argument("--index_directory", type=str)
        parser.add_argument("--index_dataset", type=str)
        parser.add_argument("--n_candidates", type=int)
        return parent_parser
