import pytest
import scipy.sparse
import numpy as np
import copy

from rasa.utils.tensorflow import model_data_utils
from rasa.utils.features import Features
from rasa.nlu.constants import INTENT
from rasa.utils.tensorflow.constants import SENTENCE


def test_create_zero_features():
    # DENSE FEATURES
    dense_feature_sentence = np.zeros((1, 100))
    dense_feature_sentence[0, 50] = 1
    dense_feature_sentence_features = Features(
        features=dense_feature_sentence,
        attribute=INTENT,
        feature_type=SENTENCE,
        origin=[],
    )
    features = [[None, None, [dense_feature_sentence_features]]]

    zero_features = model_data_utils.create_zero_features(features)
    assert len(zero_features) == 1
    assert zero_features[0].is_dense()
    assert (zero_features[0].features == np.zeros((1, 100))).all()

    # SPARSE FEATURES
    sparse_feature_sentence = scipy.sparse.coo_matrix(dense_feature_sentence)
    sparse_feature_sentence_features = Features(
        features=sparse_feature_sentence,
        attribute=INTENT,
        feature_type=SENTENCE,
        origin=[],
    )
    features = [[None, None, [sparse_feature_sentence_features]]]
    zero_features = model_data_utils.create_zero_features(features)
    assert len(zero_features) == 1
    assert zero_features[0].is_sparse()
    assert (zero_features[0].features != scipy.sparse.coo_matrix((1, 100))).nnz == 0


def test_map_tracker_features():
    zero_features = np.zeros((1, 100))
    zero_features_as_features = Features(
        features=zero_features, attribute=INTENT, feature_type=SENTENCE, origin=[]
    )
    # create zero features
    zero_features_list = [zero_features_as_features]

    # create tracker state features by setting a random index in the array to 1
    random_inds = np.random.randint(100, size=6)
    list_of_features = []
    for idx in random_inds:
        current_features = copy.deepcopy(zero_features_as_features)
        current_features.features[0, idx] = 1
        list_of_features.append([current_features])

    # organize the created features into lists ~ dialog history
    tracker_features = [
        [list_of_features[0], None, list_of_features[1]],
        [None, None, list_of_features[2]],
        [list_of_features[3], list_of_features[4], list_of_features[5]],
    ]

    (
        attribute_masks,
        dense_features,
        sparse_features,
    ) = model_data_utils.map_tracker_features(tracker_features, zero_features_list)
    expected_mask = np.array([[1, 0, 1], [0, 0, 1], [1, 1, 1]])

    assert np.all(np.squeeze(np.array(attribute_masks), 2) == expected_mask)
    assert np.array(dense_features["sentence"]).shape[-1] == zero_features.shape[-1]