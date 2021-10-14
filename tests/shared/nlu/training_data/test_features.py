import itertools
from typing import Text

import numpy as np
import pytest
import scipy.sparse

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, TEXT, FEATURE_TYPE_SEQUENCE


@pytest.mark.parametrize(
    "type,is_sparse,",
    itertools.product([FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE], [True, False]),
)
def test_print(type: Text, is_sparse: bool):
    first_dim = 1 if type == FEATURE_TYPE_SEQUENCE else 3
    matrix = np.full(shape=(first_dim, 2), fill_value=1)
    if is_sparse:
        matrix = scipy.sparse.coo_matrix(matrix)
    feat = Features(
        features=matrix,
        attribute="fixed-attribute",
        feature_type=type,
        origin="origin--doesn't-matter-here",
    )
    assert repr(feat)
    assert str(feat)


def test_combine_with_existing_dense_features():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = Features(
        np.array([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = np.array([[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]])

    existing_features.combine_with_features(new_features)

    assert np.all(expected_features == existing_features.features)


def test_combine_with_existing_dense_features_shape_mismatch():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = Features(np.array([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin")

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_combine_with_existing_sparse_features():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = Features(
        scipy.sparse.csr_matrix([[1, 0], [0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    existing_features.combine_with_features(new_features)
    actual_features = existing_features.features.toarray()

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_sparse_features_shape_mismatch():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = Features(
        scipy.sparse.csr_matrix([[0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "origin"
    )

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)
