import json
from pathlib import Path
import psutil

from autofaiss import build_index
from faiss import read_index
import numpy as np
import pytorch_lightning as pl
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW

# from torch.optim import SGD
from torchmetrics.functional import retrieval_reciprocal_rank
from tqdm import tqdm

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
            self.data = data
            data.tokenizer = self.retriever_tokenizer

        self.seed = self.hparams.seed

        # Loss
        self.triplet_loss = torch.nn.TripletMarginLoss()

        # Metrics
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

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
        query_base_ids = [query_id.rsplit("_", 1)[0] for query_id in batch["ids"]]
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

    def on_train_epoch_end(self):
        # Print newline to save printed progress per epoch
        print()

    def training_step(self, batch, batch_idx):
        anchor = self.retriever_encoder(batch["anchor"])["sentence_embedding"]
        positive = self.retriever_encoder(batch["positive"])["sentence_embedding"]
        negative = self.retriever_encoder(batch["negative"])["sentence_embedding"]

        loss = self.triplet_loss(anchor, positive, negative)

        self.log("train_loss", loss)
        return loss

    def generate_index(self):
        # Compute embeddings
        all_embeddings = []
        self.ids_labels = []

        for batch_idx, batch in enumerate(
            tqdm(self.data.index_dataloader(), desc="Calculating embeddings")
        ):
            # Get input_ids and attention_mask into device
            x = {k: v.to(self.device) for k, v in batch["context_tokenized"].items()}

            # Compute dialogue embeddings
            embeddings = self.retriever_encoder(x)["sentence_embedding"]

            # Add embeddings to cache
            all_embeddings.append(embeddings.cpu().numpy())

            # Update ids_labels array
            self.ids_labels.extend(batch["ids"])

        all_embeddings = np.concatenate(all_embeddings)

        # Generate index
        current_memory_available = f"{psutil.virtual_memory().available * 2**-30:.0f}G"

        self.index, index_infos = build_index(
            embeddings=all_embeddings,
            max_index_memory_usage="32G",
            current_memory_available=current_memory_available,
            metric_type="ip",
            save_on_disk=False,
        )

        self.index_dataset = self.data.train_data

    def on_validation_epoch_start(self):
        self.generate_index()

    def validation_step(self, batch, batch_idx):
        candidates = self.forward(batch)
        texts = [
            sample_candidates[0]["dialogue"]["text"] for sample_candidates in candidates
        ]

        # Get truth and model answers and remove "SYSTEM: "
        start = len("SYSTEM: ")
        truth_answers = [answer[start:] for answer in batch["answers"]]
        model_answers = [text.rsplit("\n", 1)[1][start:] for text in texts]

        # Compute metrics
        rouge_score = [
            self.rouge.score(truth_answer, model_answer)
            for truth_answer, model_answer in zip(truth_answers, model_answers)
        ]
        rouge_score = {
            k: [sample_score[k].fmeasure for sample_score in rouge_score]
            for k in rouge_score[0].keys()
        }

        return {
            "ids": batch["ids"],
            "truth_answers": truth_answers,
            "model_answers": model_answers,
            "metrics": rouge_score,
        }

    def validation_epoch_end(self, outs):
        # Compute average of metrics
        metrics = {
            m: round(np.concatenate([step["metrics"][m] for step in outs]).mean(), 4)
            for m in outs[0]["metrics"]
        }

        self.log_dict(metrics, prog_bar=True)

    def on_validation_epoch_end(self):
        # Print newline to save printed progress after every validation
        print()

    def on_test_epoch_start(self):
        self.generate_index()

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
        rouge_score = [
            self.rouge.score(truth_answer, model_answer)
            for truth_answer, model_answer in zip(truth_answers, model_answers)
        ]
        rouge_score = {
            k: [sample_score[k].fmeasure for sample_score in rouge_score]
            for k in rouge_score[0].keys()
        }

        return {
            "ids": batch["ids"],
            "truth_answers": truth_answers,
            "model_answers": model_answers,
            "metrics": rouge_score,
        }

    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
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
        parser.add_argument("--n_candidates", type=int, required=True)
        return parent_parser
