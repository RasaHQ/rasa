import numpy as np
import pytest

from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.constants import TEXT_ATTRIBUTE, DENSE_FEATURE_NAMES, TOKENS_NAMES
from rasa.nlu.training_data import Message
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer


def test_convert_featurizer():
    component_config = {"name": "ConveRTFeaturizer", "return_sequence": False}
    featurizer = ConveRTFeaturizer.create(component_config, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    tokens = WhitespaceTokenizer().tokenize(sentence)
    message.set(TOKENS_NAMES[TEXT_ATTRIBUTE], tokens)

    featurizer.process(message)

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE])[0]

    expected = np.array([1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353])

    assert np.allclose(vecs[:5], expected, atol=1e-5)


def test_convert_featurizer_return_sequence():
    component_config = {"name": "ConveRTFeaturizer", "return_sequence": True}
    featurizer = ConveRTFeaturizer.create(component_config, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    tokens = ConveRTTokenizer({"use_cls_token": True}).tokenize_using_convert(sentence)
    message.set(TOKENS_NAMES[TEXT_ATTRIBUTE], tokens)

    featurizer.process(message)

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE])[0]

    expected = np.array([[2.1250272, -0.6087107, -0.9836049, -0.41922584, -1.2834225]])

    assert np.allclose(vecs[:5], expected, atol=1e-5)


@pytest.mark.parametrize(
    "sentence, expected_text",
    [
        ("hello", "hello"),
        ("you're", "you re"),
        ("r. n. b.", "r n b"),
        ("rock & roll", "rock roll"),
        ("ńöñàśçií", "ńöñàśçií"),
    ],
)
def test_convert_featurizer_tokens_to_text(sentence, expected_text):
    message = Message(sentence)
    tokens = ConveRTTokenizer({"use_cls_token": False}).tokenize_using_convert(sentence)
    message.set(TOKENS_NAMES[TEXT_ATTRIBUTE], tokens)

    actual_text = ConveRTFeaturizer._tokens_to_text([message])[0]

    assert expected_text == actual_text
