from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


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
