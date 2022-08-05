import torch


class RankMetric:
    def __init__(self, d=2, length=None):
        self.d = d
        self.length = length
        if length is not None:
            self.max_value = self.calculate_max(length)
        else:
            self.max_value = None

    def __call__(self, predictions, references, mean=True):
        if self.max_value is None:
            max_value = self.calculate_max(predictions.size(-1))
        else:
            max_value = self.max_value

        predictions = torch.argsort(predictions, dim=-1)
        references = torch.argsort(references, dim=-1)

        score = 1 - (predictions - references).abs().pow(self.d).sum(dim=-1) / max_value

        if mean:
            score = score.mean()

        return score

    def calculate_max(self, length):
        return 2 * sum(i ** self.d for i in range(length - 1, 0, -2))


if __name__ == "__main__":
    # torch.manual_seed(42)

    d = 2
    L = 10
    metric = RankMetric(d, L)

    predictions = torch.rand((2, L))
    references = torch.rand((2, L))

    print("predictions", predictions, sep="\n")
    print("references", references, sep="\n")

    score = metric(predictions, references)
    print("random", score.item(), sep="\t")

    score = metric(predictions, predictions)
    print("best", score.item(), sep="\t")

    score = metric(predictions, predictions.flip(-1))
    print("worst", score.item(), sep="\t")
