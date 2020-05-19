from typing import Optional, Text, List

import pytest
import numpy as np
import scipy.sparse

from rasa.nlu.featurizers.featurizer import Features
from rasa.nlu.constants import TEXT
from rasa.nlu.training_data import Message


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_features",
    [
        (None, TEXT, [], None),
        ([Features(np.array([1, 1, 0]), TEXT, "test")], TEXT, [], [1, 1, 0]),
        (
            [
                Features(np.array([1, 1, 0]), TEXT, "c2"),
                Features(np.array([1, 2, 2]), TEXT, "c1"),
                Features(np.array([1, 2, 1]), TEXT, "c1"),
            ],
            TEXT,
            [],
            [1, 2, 1, 1, 1, 0],
        ),
        (
            [
                Features(np.array([1, 1, 0]), TEXT, "c1"),
                Features(np.array([1, 2, 1]), TEXT, "test"),
                Features(np.array([1, 1, 1]), TEXT, "test"),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
        ),
    ],
)
def test_get_dense_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_features: Optional[List[Features]],
):

    message = Message("This is a test sentence.", features=features)

    actual_features = message.get_dense_features(attribute, featurizers)

    assert np.all(actual_features == expected_features)


@pytest.mark.parametrize(
    "features, attribute, featurizers, expected_features",
    [
        (None, TEXT, [], None),
        (
            [Features(scipy.sparse.csr_matrix([1, 1, 0]), TEXT, "test")],
            TEXT,
            [],
            [1, 1, 0],
        ),
        (
            [
                Features(scipy.sparse.csr_matrix([1, 1, 0]), TEXT, "c2"),
                Features(scipy.sparse.csr_matrix([1, 2, 2]), TEXT, "c1"),
                Features(scipy.sparse.csr_matrix([1, 2, 1]), TEXT, "c1"),
            ],
            TEXT,
            [],
            [1, 2, 1, 1, 1, 0],
        ),
        (
            [
                Features(scipy.sparse.csr_matrix([1, 1, 0]), TEXT, "c1"),
                Features(scipy.sparse.csr_matrix([1, 2, 1]), TEXT, "test"),
                Features(scipy.sparse.csr_matrix([1, 1, 1]), TEXT, "test"),
            ],
            TEXT,
            ["c1"],
            [1, 1, 0],
        ),
    ],
)
def test_get_sparse_features(
    features: Optional[List[Features]],
    attribute: Text,
    featurizers: List[Text],
    expected_features: Optional[List[Features]],
):

    message = Message("This is a test sentence.", features=features)

    actual_features, actual_sen_features = message.get_sparse_features(
        attribute, featurizers
    )

    if expected_features is None:
        assert actual_features is None
    else:
        assert np.all(actual_features.toarray() == expected_features)
