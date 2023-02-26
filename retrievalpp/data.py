from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


def vary_context_length(samples, args):
    # Expand each dialogue into multiple subdialogues (varying context length)
    keys = samples.keys()
    expanded_samples = {k: [] for k in keys}

    for sample_values in zip(*samples.values()):
        sample = dict(zip(keys, sample_values))

        sample_id = sample.pop("id")
        context = sample.pop("context")
        response = sample.pop("response")

        name, start_end = sample_id.rsplit("_", 1)
        start, end = map(int, start_end.split("-"))
        length = min(len(context), args.max_nturns)
        # TODO: check if max_nturns or max_nturns+1

        for i in range(length, 0, -1):
            expanded_samples["id"].append(f"{name}_{end-i}-{end}")
            expanded_samples["context"].append(context[-i:])
            expanded_samples["response"].append(response)

            for k, v in sample:
                expanded_samples[k].append(v)

    return expanded_samples


def build_samples(samples, indices, args, tokenizer):
    # Select data
    contexts = [" ".join(turns) for turns in samples["context"]]
    answers = (
        samples["delexicalized"] if "delexicalized" in samples else samples["response"]
    )

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
    overflow = {
        "overflow": [
            input_length == args.max_length or output_length == args.max_length
            for input_length, output_length in zip(
                context_inputs["length"], answer_inputs["length"]
            )
        ]
    }

    possible_overflow = sum(overflow["overflow"])
    if possible_overflow:
        print(
            f"WARNING: Possible overflow in {possible_overflow} out of {len(samples['id'])} samples."
        )

    return {
        "idx": indices,
        "answer": answers,
        "input_ids": context_inputs["input_ids"],
        "attention_mask": context_inputs["attention_mask"],
        "answer_input_ids": answer_inputs["input_ids"],
        "answer_attention_mask": answer_inputs["attention_mask"],
    } | overflow


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
