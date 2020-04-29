import logging
import pytest

from transformers import BertJapaneseTokenizer
from rasa.nlu.utils.hugging_face.hf_transformers import HFTransformersNLP


def test_tokenizer_when_selected_japanese_model_weights():
    transformers_config = {"model_name": "bert", "model_weights": "bert-base-japanese-whole-word-masking"}
    nlp = HFTransformersNLP(transformers_config)
    assert nlp.tokenizer == BertJapaneseTokenizer
