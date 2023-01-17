import torch
from transformers import AutoConfig, AutoModel, PreTrainedModel, PretrainedConfig


class EncoderConfig(PretrainedConfig):
    model_type = "encodermodel"

    def __init__(
        self, base_model="sentence-transformers/multi-qa-mpnet-base-dot-v1", **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = base_model


class EncoderModel(PreTrainedModel):
    config_class = EncoderConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize model
        self.model = AutoModel.from_pretrained(config.base_model)

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


class RetrievalConfig(PretrainedConfig):
    model_type = "retrievalmodel"

    def __init__(
        self,
        base_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        dual=False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = base_model
        self.dual = dual


class RetrievalModel(PreTrainedModel):
    config_class = RetrievalConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize models
        config_question = EncoderConfig(config.base_model)
        config_answer = EncoderConfig(config.base_model)
        self.encoder_question = EncoderModel(config_question)
        self.encoder_answer = (
            EncoderModel(config_answer) if self.config.dual else self.encoder_question
        )

    def forward(self, mode="question", **inputs):
        return (
            self.encoder_question(**inputs)
            if mode == "question"
            else self.encoder_answer(**inputs)
        )
