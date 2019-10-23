import numpy as np
import scipy.sparse

from rasa.nlu.featurizers.featurzier import Featurizer
from rasa.nlu.constants import (
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_VECTOR_SPARSE_FEATURE_NAMES,
    MESSAGE_TEXT_ATTRIBUTE,
)
from rasa.nlu.training_data import Message


def test_combine_with_existing_dense_features():

    featurizer = Featurizer()
    attribute = MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]

    existing_features = [[1, 0, 2, 3], [2, 0, 0, 1]]
    new_features = [[1, 0], [0, 1]]
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    actual_features = featurizer._combine_with_existing_dense_features(
        message, new_features, attribute
    )

    assert np.all(expected_features == actual_features)


def test_combine_with_existing_sparse_features():

    featurizer = Featurizer()
    attribute = MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]

    existing_features = scipy.sparse.csr_matrix([[1, 0, 2, 3], [2, 0, 0, 1]])
    new_features = scipy.sparse.csr_matrix([[1, 0], [0, 1]])
    expected_features = [[1, 0, 2, 3, 1, 0], [2, 0, 0, 1, 0, 1]]

    message = Message("This is a text.")
    message.set(attribute, existing_features)

    actual_features = featurizer._combine_with_existing_sparse_features(
        message, new_features, attribute
    )
    actual_features = actual_features.toarray()

    assert np.all(expected_features == actual_features)
