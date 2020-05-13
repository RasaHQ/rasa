import numpy as np
import pytest
import scipy.sparse

from rasa.nlu.featurizers.featurizer import DenseFeaturizer, Features
from rasa.nlu.constants import TEXT, FEATURE_TYPE_SEQUENCE


def test_combine_with_existing_dense_features():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = np.array([[1, 0], [0, 1]])
    expected_features = np.array([[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]])

    actual_features = existing_features.combine_with_features(new_features)

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_dense_features_shape_mismatch():
    existing_features = Features(
        np.array([[1, 0, 2, 3], [2, 0, 0, 1]]), FEATURE_TYPE_SEQUENCE, TEXT, "test"
    )
    new_features = np.array([[0, 1]])

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


def test_combine_with_existing_sparse_features():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = scipy.sparse.csr_matrix([[1, 0], [0, 1]])
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    actual_features = existing_features.combine_with_features(new_features)
    actual_features = actual_features.toarray()

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_sparse_features_shape_mismatch():
    existing_features = Features(
        scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]]),
        FEATURE_TYPE_SEQUENCE,
        TEXT,
        "test",
    )
    new_features = scipy.sparse.csr_matrix([[0, 1]])

    with pytest.raises(ValueError):
        existing_features.combine_with_features(new_features)


@pytest.mark.parametrize(
    "pooling, features, expected",
    [
        (
            "mean",
            np.array([[0.5, 3, 0.4, 0.1], [0, 0, 0, 0], [0.5, 3, 0.4, 0.1]]),
            np.array([[0.5, 3, 0.4, 0.1]]),
        ),
        (
            "max",
            np.array([[1.0, 3.0, 0.0, 2.0], [4.0, 3.0, 1.0, 0.0]]),
            np.array([[4.0, 3.0, 1.0, 2.0]]),
        ),
        (
            "max",
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_cls_vector(pooling, features, expected):
    actual = DenseFeaturizer._calculate_cls_vector(features, pooling)

    assert np.all(actual == expected)
