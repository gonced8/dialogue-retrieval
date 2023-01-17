from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def build_samples(samples, indices, args, tokenizer):
    # Select data
    contexts = [" ".join(turns[:-1]) for turns in samples["text"]]
    answers = [turns[-1] for turns in samples["text"]]

    # Tokenize
    context_inputs = tokenizer(
        contexts,
        max_length=args.max_length,
        truncation=True,
        return_length=True,
    )
    answer_inputs = tokenizer(
        answers,
        max_length=args.max_length,
        truncation=True,
        return_length=True,
    )

    # Verify if possible truncation
    if any(sample_len == args.max_length for sample_len in context_inputs["length"]):
        print("WARNING: Possible truncation occurring in input_ids.")

    if any(sample_len == args.max_length for sample_len in answer_inputs["length"]):
        print("WARNING: Possible truncation occurring in answer_input_ids.")

    return {
        "idx": indices,
        "answer": answers,
        "input_ids": context_inputs["input_ids"],
        "attention_mask": context_inputs["attention_mask"],
        "answer_input_ids": answer_inputs["input_ids"],
        "answer_attention_mask": answer_inputs["attention_mask"],
    }


@dataclass
class RetrievalDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        answer_features = [
            {
                "input_ids": sample.pop("answer_input_ids"),
                "attention_mask": sample.pop("answer_attention_mask"),
            }
            for sample in features
        ]

        tensor_features = [
            {k: v for k, v in sample.items() if torch.is_tensor(v)}
            for sample in features
        ]

        text_features = {
            k: [sample[k] for sample in features]
            for k, v in features[0].items()
            if isinstance(v, str)
        }

        batch = self.tokenizer.pad(
            tensor_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch_answer = self.tokenizer.pad(
            answer_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Add answer features
        batch["answer_input_ids"] = batch_answer["input_ids"]
        batch["answer_attention_mask"] = batch_answer["attention_mask"]

        # Add text features
        batch.update(text_features)

        return batch
