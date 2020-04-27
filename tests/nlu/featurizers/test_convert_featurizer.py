import numpy as np
import pytest

from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.training_data import TrainingData
from rasa.nlu.tokenizers.convert_tokenizer import ConveRTTokenizer
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, TOKENS_NAMES, RESPONSE, INTENT
from rasa.nlu.training_data import Message
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer


def test_convert_featurizer_process():
    featurizer = ConveRTFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    tokens = ConveRTTokenizer().tokenize(message, attribute=TEXT)
    tokens = Tokenizer.add_cls_token(tokens, attribute=TEXT)
    message.set(TOKENS_NAMES[TEXT], tokens)

    featurizer.process(message)

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT])

    assert len(tokens) == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)


def test_convert_featurizer_train():
    featurizer = ConveRTFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today ?"
    message = Message(sentence)
    message.set(RESPONSE, sentence)
    tokens = ConveRTTokenizer().tokenize(message, attribute=TEXT)
    tokens = Tokenizer.add_cls_token(tokens, attribute=TEXT)
    message.set(TOKENS_NAMES[TEXT], tokens)
    message.set(TOKENS_NAMES[RESPONSE], tokens)

    featurizer.train(TrainingData([message]), RasaNLUModelConfig())

    expected = np.array([2.2636216, -0.26475656, -1.1358104, -0.49751878, -1.3946456])
    expected_cls = np.array(
        [1.0251294, -0.04053932, -0.7018805, -0.82054937, -0.75054353]
    )

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT])

    assert len(tokens) == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)

    vecs = message.get(DENSE_FEATURE_NAMES[RESPONSE])

    assert len(tokens) == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)

    vecs = message.get(DENSE_FEATURE_NAMES[INTENT])

    assert vecs is None


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
    tokens = ConveRTTokenizer().tokenize(Message(sentence), attribute=TEXT)

    actual_text = ConveRTFeaturizer._tokens_to_text([tokens])[0]

    assert expected_text == actual_text


def test_align_token_features():
    tokens = ConveRTTokenizer().tokenize(Message("In Aarhus and Ahaus"), attribute=TEXT)
    x = sum(t.get("number_of_sub_words") for t in tokens)
    token_features = np.random.rand(1, x, 64)

    actual_features = ConveRTFeaturizer._align_token_features([tokens], token_features)

    assert np.all(actual_features[0][0] == token_features[0][0])
    assert np.all(actual_features[0][1] == np.mean(token_features[0][1:3]))
    assert np.all(actual_features[0][2] == token_features[0][3])
    assert np.all(actual_features[0][3] == np.mean(token_features[0][4:6]))
