# -*- coding: utf-8 -
import numpy as np
import pytest

from rasa.nlu import config, train
from rasa.nlu import training_data
from rasa.nlu.tokenizers import Token
from rasa.nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import ComponentBuilder

CONFIG_DEFAULTS_PATH = "sample_configs/config_defaults.yml"


def test_regex_featurizer_with_jieba_tokenizer(sentence, expected, labeled_tokens):
    from rasa.nlu.featurizers.regex_featurizer import RegexFeaturizer
    from rasa.nlu.tokenizers.jieba_tokenizer import JiebaTokenizer

    patterns = [
        {"pattern": "南京", "name": "pattern_0", "usage": "intent"},
        {"pattern": "北京", "name": "pattern_1", "usage": "intent"},
        {"pattern": "上海", "name": "pattern_1", "usage": "intent"},
    ]
    ftr = RegexFeaturizer(known_patterns=patterns)

    tokenizer = JiebaTokenizer()

    message = Message(sentence)
    tokenizer.process(message)

    result = ftr.features_for_patterns(message)
    assert np.allclose(result, expected, atol=1e-10)

    # the tokenizer should have added tokens
    assert len(message.get("tokens", [])) > 0
    # the number of regex matches on each token should match
    for t in message.get("tokens"):
        print(t.text)
    for i, token in enumerate(message.get("tokens")):
        token_matches = token.get("pattern").values()
        print(token.get("pattern").values())
        num_matches = sum(token_matches)
        print(num_matches)
        print(labeled_tokens)
        print(labeled_tokens.count(i))
        assert num_matches == labeled_tokens.count(i)


test_regex_featurizer_with_jieba_tokenizer(
    "买一张从南京到上海的火车票", [1.0, 0.0, 1.0], [4.0, 5.0, 7.0, 8.0]
)
