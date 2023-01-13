from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


@dataclass
class DataCollatorWithPaddingAndText:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        # Add text features
        batch.update(text_features)

        return batch
