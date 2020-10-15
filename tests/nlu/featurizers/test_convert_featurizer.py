import numpy as np
import pytest

from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, NUMBER_OF_SUB_TOKENS
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.convert_featurizer import ConveRTFeaturizer


# TODO
#   skip tests as the ConveRT model is not publicly available anymore (see
#   https://github.com/RasaHQ/rasa/issues/6806)


@pytest.mark.skip
def test_convert_featurizer_process(component_builder):
    tokenizer = WhitespaceTokenizer()
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)
    sentence = "Hey how are you today ?"
    message = Message.build(text=sentence)

    td = TrainingData([message])
    tokenizer.train(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    featurizer.process(message, tf_hub_module=featurizer.module)

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


@pytest.mark.skip
def test_convert_featurizer_train(component_builder):
    tokenizer = WhitespaceTokenizer()
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)

    sentence = "Hey how are you today ?"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)

    td = TrainingData([message])
    tokenizer.train(td)

    tokens = featurizer.tokenize(message, attribute=TEXT)

    message.set(TOKENS_NAMES[TEXT], tokens)
    message.set(TOKENS_NAMES[RESPONSE], tokens)

    featurizer.train(
        TrainingData([message]), RasaNLUModelConfig(), tf_hub_module=featurizer.module
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
@pytest.mark.skip
def test_convert_featurizer_tokens_to_text(component_builder, sentence, expected_text):
    tokenizer = WhitespaceTokenizer()
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)
    message = Message.build(text=sentence)
    td = TrainingData([message])
    tokenizer.train(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    actual_text = ConveRTFeaturizer._tokens_to_text([tokens])[0]

    assert expected_text == actual_text


@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "forecast for lunch",
            ["forecast", "for", "lunch"],
            [(0, 8), (9, 12), (13, 18)],
        ),
        ("hello", ["hello"], [(0, 5)]),
        ("you're", ["you", "re"], [(0, 3), (4, 6)]),
        ("r. n. b.", ["r", "n", "b"], [(0, 1), (3, 4), (6, 7)]),
        ("rock & roll", ["rock", "&", "roll"], [(0, 4), (5, 6), (7, 11)]),
        ("ńöñàśçií", ["ńöñàśçií"], [(0, 8)]),
    ],
)
@pytest.mark.skip
def test_convert_featurizer_token_edge_cases(
    component_builder, text, expected_tokens, expected_indices
):
    tokenizer = WhitespaceTokenizer()
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)
    message = Message.build(text=text)
    td = TrainingData([message])
    tokenizer.train(td)
    tokens = featurizer.tokenize(message, attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]


@pytest.mark.parametrize(
    "text, expected_number_of_sub_tokens",
    [("Aarhus is a city", [2, 1, 1, 1]), ("sentence embeddings", [1, 3])],
)
@pytest.mark.skip
def test_convert_featurizer_number_of_sub_tokens(
    component_builder, text, expected_number_of_sub_tokens
):
    tokenizer = WhitespaceTokenizer()
    featurizer = component_builder.create_component_from_class(ConveRTFeaturizer)

    message = Message.build(text=text)
    td = TrainingData([message])
    tokenizer.train(td)

    tokens = featurizer.tokenize(message, attribute=TEXT)

    assert [
        t.get(NUMBER_OF_SUB_TOKENS) for t in tokens
    ] == expected_number_of_sub_tokens
