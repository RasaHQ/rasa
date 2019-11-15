import numpy as np
import pytest
import scipy.sparse

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
        assert isinstance(o, scipy.sparse.coo_matrix)
        assert o.data[0] == 1
        assert o.col[0] == i
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
