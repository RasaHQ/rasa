from transformers import (
    TFBertModel,
    TFOpenAIGPTModel,
    TFGPT2Model,
    TFXLNetModel,
    TFXLMModel,
    TFDistilBertModel,
    TFRobertaModel,
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
)
from rasa.nlu.utils.hugging_face.transformers_pre_post_processors import *

model_class_dict = {
    "bert": TFBertModel,
    "openaigpt": TFOpenAIGPTModel,
    "gpt2": TFGPT2Model,
    "xlnet": TFXLNetModel,
    "xlm": TFXLMModel,
    "distilbert": TFDistilBertModel,
    "roberta": TFRobertaModel,
}
model_tokenizer_dict = {
    "bert": BertTokenizer,
    "openaigpt": OpenAIGPTTokenizer,
    "gpt2": GPT2Tokenizer,
    "xlnet": XLNetTokenizer,
    "xlm": XLMTokenizer,
    "distilbert": DistilBertTokenizer,
    "roberta": RobertaTokenizer,
}
model_weights_defaults = {
    "bert": "bert-base-uncased",
    "openai-gpt": "openai-gpt",
    "gpt2": "gpt2",
    "xlnet": "xlnet-base-cased",
    "xlm": "xlm-mlm-enfr-1024",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
}

model_special_tokens_pre_processors = {
    "bert": bert_tokens_pre_processor,
    "openai-gpt": gpt_tokens_pre_processor,
    "gpt2": gpt_tokens_pre_processor,
    "xlnet": xlnet_tokens_pre_processor,
    "xlm": xlm_tokens_pre_processor,
    "distilbert": bert_tokens_pre_processor,
    "roberta": roberta_tokens_pre_processor,
}

model_embeddings_post_processors = {
    "bert": bert_embeddings_post_process,
    "openai-gpt": gpt_embeddings_post_process,
    "gpt2": gpt_embeddings_post_process,
    "xlnet": xlnet_embeddings_post_process,
    "xlm": xlm_embeddings_post_process,
    "distilbert": bert_embeddings_post_process,
    "roberta": roberta_embeddings_post_process,
}
