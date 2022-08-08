from torchmetrics.functional import RetrievalNormalizedDCG


class MinMax_NCDG(RetrievalNormalizedDCG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _metric(self, preds: Tensor, target: Tensor) -> Tensor:
        # TODO: normalize
        print(preds.shape, target.shape)
        return retrieval_normalized_dcg(preds, target, k=self.k)
