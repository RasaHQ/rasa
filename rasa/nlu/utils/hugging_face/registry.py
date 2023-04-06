import logging
from typing import Dict, Text, Type

# Explicitly set logging level for this module before any import
# because otherwise it logs tensorflow/pytorch versions
logging.getLogger("transformers.file_utils").setLevel(logging.WARNING)

from transformers import (  # noqa: E402
    TFPreTrainedModel,
    TFBertModel,
    TFOpenAIGPTModel,
    TFGPT2Model,
    TFXLNetModel,
    # TFXLMModel,
    TFDistilBertModel,
    TFRobertaModel,
    TFCamembertModel,
    PreTrainedTokenizer,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    XLNetTokenizer,
    # XLMTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    CamembertTokenizer,
)
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import (  # noqa: E402, E501
    bert_tokens_pre_processor,
    gpt_tokens_pre_processor,
    xlnet_tokens_pre_processor,
    roberta_tokens_pre_processor,
    bert_embeddings_post_processor,
    gpt_embeddings_post_processor,
    xlnet_embeddings_post_processor,
    roberta_embeddings_post_processor,
    bert_tokens_cleaner,
    openaigpt_tokens_cleaner,
    gpt2_tokens_cleaner,
    xlnet_tokens_cleaner,
    camembert_tokens_pre_processor,
)


model_class_dict: Dict[Text, Type[TFPreTrainedModel]] = {
    "bert": TFBertModel,
    "gpt": TFOpenAIGPTModel,
    "gpt2": TFGPT2Model,
    "xlnet": TFXLNetModel,
    # "xlm": TFXLMModel, # Currently doesn't work because of a bug in transformers
    # library https://github.com/huggingface/transformers/issues/2729
    "distilbert": TFDistilBertModel,
    "roberta": TFRobertaModel,
    "camembert": TFCamembertModel,
}
model_tokenizer_dict: Dict[Text, Type[PreTrainedTokenizer]] = {
    "bert": BertTokenizer,
    "gpt": OpenAIGPTTokenizer,
    "gpt2": GPT2Tokenizer,
    "xlnet": XLNetTokenizer,
    # "xlm": XLMTokenizer,
    "distilbert": DistilBertTokenizer,
    "roberta": RobertaTokenizer,
    "camembert": CamembertTokenizer,
}
model_weights_defaults = {
    "bert": "rasa/LaBSE",
    "gpt": "openai-gpt",
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    # "xlm": "xlm-mlm-enfr-1024",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "camembert": "camembert-base",
}

model_special_tokens_pre_processors = {
    "bert": bert_tokens_pre_processor,
    "gpt": gpt_tokens_pre_processor,
    "gpt2": gpt_tokens_pre_processor,
    "xlnet": xlnet_tokens_pre_processor,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_pre_processor,
    "roberta": roberta_tokens_pre_processor,
    "camembert": camembert_tokens_pre_processor,
}

model_tokens_cleaners = {
    "bert": bert_tokens_cleaner,
    "gpt": openaigpt_tokens_cleaner,
    "gpt2": gpt2_tokens_cleaner,
    "xlnet": xlnet_tokens_cleaner,
    # "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_cleaner,  # uses the same as BERT
    "roberta": gpt2_tokens_cleaner,  # Uses the same as GPT2
    "camembert": xlnet_tokens_cleaner,  # Removing underscores _
}

model_embeddings_post_processors = {
    "bert": bert_embeddings_post_processor,
    "gpt": gpt_embeddings_post_processor,
    "gpt2": gpt_embeddings_post_processor,
    "xlnet": xlnet_embeddings_post_processor,
    # "xlm": xlm_embeddings_post_processor,
    "distilbert": bert_embeddings_post_processor,
    "roberta": roberta_embeddings_post_processor,
    "camembert": roberta_embeddings_post_processor,
}
