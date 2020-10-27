import numpy as np
import pytest

from rasa.nlu.tokenizers.convert_tokenizer import (
    ConveRTTokenizer,
    RESTRICTED_ACCESS_URL,
)
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer


def test_convert_featurizer_process(component_builder):

    component_config = {"name": "ConveRTTokenizer", "model_url": RESTRICTED_ACCESS_URL}
    tokenizer = ConveRTTokenizer(component_config)
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)

    sentence = "Hey how are you today ?"
    message = Message(data={TEXT: sentence})
    tokens = tokenizer.tokenize(message, attribute=TEXT)
    message.set(TOKENS_NAMES[TEXT], tokens)

    featurizer.process(message, tf_hub_module=tokenizer.module)

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    seq_vecs, sent_vecs = message.get_dense_features(TEXT, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)


def test_convert_featurizer_train(component_builder):
    component_config = {"name": "ConveRTTokenizer", "model_url": RESTRICTED_ACCESS_URL}
    tokenizer = ConveRTTokenizer(component_config)
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)

    sentence = "Hey how are you today ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)

    tokens = tokenizer.tokenize(message, attribute=TEXT)

    message.set(TOKENS_NAMES[TEXT], tokens)
    message.set(TOKENS_NAMES[RESPONSE], tokens)

    featurizer.train(
        TrainingData([message]), RasaNLUModelConfig(), tf_hub_module=tokenizer.module
    )

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    seq_vecs, sent_vecs = message.get_dense_features(TEXT, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)

    seq_vecs, sent_vecs = message.get_dense_features(RESPONSE, [])

    seq_vecs = seq_vecs.features
    sent_vecs = sent_vecs.features

    assert len(tokens) == len(seq_vecs)
    assert np.allclose(seq_vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(sent_vecs[-1][:5], expected_cls, atol=1e-5)

    seq_vecs, sent_vecs = message.get_dense_features(INTENT, [])

    assert seq_vecs is None
    assert sent_vecs is None


@pytest.mark.parametrize(
    "sentence, expected_text",
    [
        ("hello", "hello"),
        ("you're", "you re"),
        ("r. n. b.", "r n b"),
        ("rock & roll", "rock & roll"),
        ("ńöñàśçií", "ńöñàśçií"),
    ],
)
def test_convert_featurizer_tokens_to_text(sentence, expected_text):
    component_config = {"name": "ConveRTTokenizer", "model_url": RESTRICTED_ACCESS_URL}
    tokenizer = ConveRTTokenizer(component_config)
    tokens = tokenizer.tokenize(Message(data={TEXT: sentence}), attribute=TEXT)

    actual_text = ConveRTFeaturizer._tokens_to_text([tokens])[0]

    assert expected_text == actual_text
