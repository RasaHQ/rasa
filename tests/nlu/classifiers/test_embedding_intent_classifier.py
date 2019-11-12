import numpy as np
import pytest

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_SPARSE_FEATURE_NAMES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_INTENT_ATTRIBUTE,
)
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.training_data import Message


def test_compute_default_label_features():
    label_features = [
        Message("test a"),
        Message("test b"),
        Message("test c"),
        Message("test d"),
    ]

    output = EmbeddingIntentClassifier._compute_default_label_features(label_features)

    output = output[0]

    assert output.size == len(label_features)
    for i, o in enumerate(output):
        assert o.data[0] == 1
        assert o.indices[0] == i
        assert o.shape == (1, len(label_features))


@pytest.mark.parametrize(
    "messages, expected",
    [
        (
            [
                Message(
                    "test a",
                    data={
                        MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[
                            MESSAGE_TEXT_ATTRIBUTE
                        ]: np.zeros(1),
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[
                            MESSAGE_TEXT_ATTRIBUTE
                        ]: np.zeros(1),
                    },
                ),
                Message(
                    "test b",
                    data={
                        MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[
                            MESSAGE_TEXT_ATTRIBUTE
                        ]: np.zeros(1),
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[
                            MESSAGE_TEXT_ATTRIBUTE
                        ]: np.zeros(1),
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
                        MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[
                            MESSAGE_INTENT_ATTRIBUTE
                        ]: np.zeros(1),
                        MESSAGE_VECTOR_DENSE_FEATURE_NAMES[
                            MESSAGE_INTENT_ATTRIBUTE
                        ]: np.zeros(1),
                    },
                )
            ],
            False,
        ),
    ],
)
def test_check_labels_features_exist(messages, expected):
    attribute = MESSAGE_TEXT_ATTRIBUTE

    assert (
        EmbeddingIntentClassifier._check_labels_features_exist(messages, attribute)
        == expected
    )
