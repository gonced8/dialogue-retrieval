from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)


class Encoder(PreTrainedModel):
    def __init__(self, model_name):
        super(Encoder, self).__init__(config=AutoConfig.from_pretrained(model_name))
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, **inputs):
        model_output = self.model(**inputs)
        return self.cls_pooling(model_output)

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
