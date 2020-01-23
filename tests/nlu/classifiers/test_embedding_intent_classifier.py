import numpy as np
import pytest
import scipy.sparse

from unittest.mock import Mock

from rasa.nlu import train
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import (
    TEXT_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    INTENT_ATTRIBUTE,
)
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.model import Interpreter
from rasa.nlu.training_data import Message
from rasa.utils import train_utils
from tests.nlu.conftest import DEFAULT_DATA_PATH


def test_compute_default_label_features():
    label_features = [
        Message("test a"),
        Message("test b"),
        Message("test c"),
        Message("test d"),
    ]

    output = EmbeddingIntentClassifier._compute_default_label_features(label_features)

    output = output[0]

    for i, o in enumerate(output):
        assert isinstance(o, np.ndarray)
        assert o[0][i] == 1
        assert o.shape == (1, len(label_features))


def test_get_num_of_features():
    session_data = {
        "text_features": [
            np.array(
                [
                    np.random.rand(5, 14),
                    np.random.rand(2, 14),
                    np.random.rand(3, 14),
                    np.random.rand(1, 14),
                    np.random.rand(3, 14),
                ]
            ),
            np.array(
                [
                    scipy.sparse.csr_matrix(np.random.randint(5, size=(5, 10))),
                    scipy.sparse.csr_matrix(np.random.randint(5, size=(2, 10))),
                    scipy.sparse.csr_matrix(np.random.randint(5, size=(3, 10))),
                    scipy.sparse.csr_matrix(np.random.randint(5, size=(1, 10))),
                    scipy.sparse.csr_matrix(np.random.randint(5, size=(3, 10))),
                ]
            ),
        ]
    }

    num_features = EmbeddingIntentClassifier._get_num_of_features(
        session_data, "text_features"
    )

    assert num_features == 24


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    "test a",
                    data={
                        SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]: np.zeros(1),
                        DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE]: np.zeros(1),
                    },
                ),
                Message(
                    "test b",
                    data={
                        SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]: np.zeros(1),
                        DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE]: np.zeros(1),
                    },
                ),
            ],
            True,
        ),
        (
            [
                Message(
                    "test a",
                    data={
                        SPARSE_FEATURE_NAMES[INTENT_ATTRIBUTE]: np.zeros(1),
                        DENSE_FEATURE_NAMES[INTENT_ATTRIBUTE]: np.zeros(1),
                    },
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(messages, expected):
    attribute = TEXT_ATTRIBUTE

    assert (
        EmbeddingIntentClassifier._check_labels_features_exist(messages, attribute)
        == expected
    )


async def test_train(component_builder, tmpdir):
    pipeline = [
        {"name": "ConveRTTokenizer"},
        {"name": "CountVectorsFeaturizer"},
        {"name": "ConveRTFeaturizer"},
        {"name": "EmbeddingIntentClassifier"},
    ]

    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": "en"})

    (trained, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=DEFAULT_DATA_PATH,
        component_builder=component_builder,
    )

    assert trained.pipeline

    loaded = Interpreter.load(persisted_path, component_builder)
    assert loaded.pipeline
    assert loaded.parse("hello") is not None
    assert loaded.parse("Hello today is Monday, again!") is not None


async def test_raise_error_on_incorrect_pipeline(component_builder, tmpdir):
    from rasa.nlu import train

    _config = RasaNLUModelConfig(
        {
            "pipeline": [
                {"name": "WhitespaceTokenizer"},
                {"name": "EmbeddingIntentClassifier"},
            ],
            "language": "en",
        }
    )

    with pytest.raises(Exception) as e:
        await train(
            _config,
            path=tmpdir.strpath,
            data=DEFAULT_DATA_PATH,
            component_builder=component_builder,
        )

    assert (
        "Failed to validate component 'EmbeddingIntentClassifier'. Missing one of "
        "the following properties: " in str(e.value)
    )


def as_pipeline(*components):
    return [{"name": c} for c in components]


@pytest.mark.parametrize(
    "classifier_params, data_path, output_length, output_should_sum_to_1",
    [
        ({"random_seed": 42}, "data/test/many_intents.md", 10, True),  # default config
        (
            {"random_seed": 42, "ranking_length": 0},
            "data/test/many_intents.md",
            LABEL_RANKING_LENGTH,
            False,
        ),  # no normalization
        (
            {"random_seed": 42, "ranking_length": 3},
            "data/test/many_intents.md",
            3,
            True,
        ),  # lower than default ranking_length
        (
            {"random_seed": 42, "ranking_length": 12},
            "data/test/many_intents.md",
            LABEL_RANKING_LENGTH,
            False,
        ),  # higher than default ranking_length
        (
            {"random_seed": 42},
            "examples/moodbot/data/nlu.md",
            7,
            True,
        ),  # less intents than default ranking_length
    ],
)
async def test_softmax_normalization(
    component_builder,
    tmpdir,
    classifier_params,
    data_path,
    output_length,
    output_should_sum_to_1,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "EmbeddingIntentClassifier"
    )
    assert pipeline[2]["name"] == "EmbeddingIntentClassifier"
    pipeline[2].update(classifier_params)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data=data_path,
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")
    # check that the output was correctly truncated after normalization
    assert len(intent_ranking) == output_length

    # check whether normalization had the expected effect
    output_sums_to_1 = sum(
        [intent.get("confidence") for intent in intent_ranking]
    ) == pytest.approx(1)
    assert output_sums_to_1 == output_should_sum_to_1

    # check whether the normalization of rankings is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]


@pytest.mark.parametrize(
    "classifier_params, output_length",
    [({"loss_type": "margin", "random_seed": 42}, LABEL_RANKING_LENGTH)],
)
async def test_margin_loss_is_not_normalized(
    monkeypatch, component_builder, tmpdir, classifier_params, output_length,
):
    pipeline = as_pipeline(
        "WhitespaceTokenizer", "CountVectorsFeaturizer", "EmbeddingIntentClassifier"
    )
    assert pipeline[2]["name"] == "EmbeddingIntentClassifier"
    pipeline[2].update(classifier_params)

    mock = Mock()
    monkeypatch.setattr(train_utils, "normalize", mock.normalize)

    _config = RasaNLUModelConfig({"pipeline": pipeline})
    (trained_model, _, persisted_path) = await train(
        _config,
        path=tmpdir.strpath,
        data="data/test/many_intents.md",
        component_builder=component_builder,
    )
    loaded = Interpreter.load(persisted_path, component_builder)

    parse_data = loaded.parse("hello")
    intent_ranking = parse_data.get("intent_ranking")

    # check that the output was not normalized
    mock.normalize.assert_not_called()

    # check that the output was correctly truncated
    assert len(intent_ranking) == output_length

    # make sure top ranking is reflected in intent prediction
    assert parse_data.get("intent") == intent_ranking[0]
