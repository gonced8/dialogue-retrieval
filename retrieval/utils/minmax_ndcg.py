import torch
from torchmetrics.functional import retrieval_normalized_dcg


def normalize_minmax(x: torch.Tensor) -> torch.Tensor:
    # Set min to 0
    x -= x.min(1, keepdim=True)[0]

    # Set max to 1 (avoiding zero division)
    max_value = x.max(1, keepdim=True)[0]
    max_value[max_value == 0] = 1.0
    x = torch.div(x, max_value)

    return x


def minmax_ndcg(
    preds: torch.Tensor, target: torch.Tensor, k: int = None
) -> torch.Tensor:
    preds = normalize_minmax(preds)
    target = normalize_minmax(target)
    return retrieval_normalized_dcg(preds, target, k=k)
