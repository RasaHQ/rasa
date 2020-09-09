from typing import Optional, Text, List

import pytest
import numpy as np
import scipy.sparse

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import (
    TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
    ACTION_NAME,
)
import rasa.shared.nlu.training_data.message
from rasa.shared.nlu.training_data.message import Message


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_seq_features, expected_sen_features",
    [
        (None, TEXT, [], None, None),
        (
            [Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "test")],
            TEXT,
            [],
            [1, 1, 0],
            None,
        ),
        (
            [
                Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c2"),
                Features(np.array([1, 2, 2]), FEATURE_TYPE_SENTENCE, TEXT, "c1"),
                Features(np.array([1, 2, 1]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            [],
            [1, 1, 0, 1, 2, 1],
            [1, 2, 2],
        ),
        (
            [
                Features(np.array([1, 1, 0]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
                Features(np.array([1, 2, 1]), FEATURE_TYPE_SENTENCE, TEXT, "test"),
                Features(np.array([1, 1, 1]), FEATURE_TYPE_SEQUENCE, TEXT, "test"),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
            None,
        ),
    ],
)
def test_get_dense_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_seq_features: Optional[List[Features]],
    expected_sen_features: Optional[List[Features]],
):

    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual_seq_features, actual_sen_features = message.get_dense_features(
        attribute, featurizers
    )
    if actual_seq_features:
        actual_seq_features = actual_seq_features.features
    if actual_sen_features:
        actual_sen_features = actual_sen_features.features

    assert np.all(actual_sen_features == expected_sen_features)
    assert np.all(actual_seq_features == expected_seq_features)


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_seq_features, expected_sen_features",
    [
        (None, TEXT, [], None, None),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                )
            ],
            TEXT,
            [],
            [1, 1, 0],
            None,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 2]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
            ],
            TEXT,
            [],
            [1, 1, 0, 1, 2, 1],
            [1, 2, 2],
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c1",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 2, 1]),
                    FEATURE_TYPE_SENTENCE,
                    TEXT,
                    "test",
                ),
                Features(
                    scipy.sparse.csr_matrix([1, 1, 1]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                ),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
            None,
        ),
    ],
)
def test_get_sparse_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_seq_features: Optional[List[Features]],
    expected_sen_features: Optional[List[Features]],
):
    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual_seq_features, actual_sen_features = message.get_sparse_features(
        attribute, featurizers
    )
    if actual_seq_features:
        actual_seq_features = actual_seq_features.features
    if actual_sen_features:
        actual_sen_features = actual_sen_features.features

    if expected_seq_features is None:
        assert actual_seq_features is None
    else:
        assert actual_seq_features is not None
        assert np.all(actual_seq_features.toarray() == expected_seq_features)

    if expected_sen_features is None:
        assert actual_sen_features is None
    else:
        assert actual_sen_features is not None
        assert np.all(actual_sen_features.toarray() == expected_sen_features)


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected",
    [
        (None, TEXT, [], False),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "test",
                )
            ],
            TEXT,
            [],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            [],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            ["c1"],
            True,
        ),
        (
            [
                Features(
                    scipy.sparse.csr_matrix([1, 1, 0]),
                    FEATURE_TYPE_SEQUENCE,
                    TEXT,
                    "c2",
                ),
                Features(np.ndarray([1, 2, 2]), FEATURE_TYPE_SEQUENCE, TEXT, "c1"),
            ],
            TEXT,
            ["other"],
            False,
        ),
    ],
)
def test_features_present(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected: bool,
):
    message = Message(data={TEXT: "This is a test sentence."}, features=features)

    actual = message.features_present(attribute, featurizers)

    assert actual == expected


def test_ordered():
    target = {"a": [1, 3, 2], "c": "a", "b": 1}
    assert rasa.shared.nlu.training_data.message.ordered(target) == [
        ("a", [1, 2, 3]),
        ("b", 1),
        ("c", "a"),
    ]


def test_build_from_action():
    test_action_name = "test_action_name"
    test_action_text = "test action text"
    assert Message.build_from_action(
        action_text=test_action_text, action_name=test_action_name
    ) == Message(data={ACTION_NAME: test_action_name, ACTION_TEXT: test_action_text})
