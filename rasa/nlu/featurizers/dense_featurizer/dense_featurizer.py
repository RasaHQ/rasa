from abc import ABC
from typing import Text

import numpy as np

from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.shared.exceptions import InvalidConfigException
from rasa.utils.tensorflow.constants import MAX_POOLING, MEAN_POOLING


class DenseFeaturizer(Featurizer[np.ndarray], ABC):
    """Base class for all dense featurizers."""

    @staticmethod
    def aggregate_sequence_features(
        dense_sequence_features: np.ndarray,
        pooling_operation: Text,
        only_non_zero_vectors: bool = True,
    ) -> np.ndarray:
        """Aggregates the non-zero vectors of a dense sequence feature matrix.

        Args:
          dense_sequence_features: a 2-dimensional matrix where the first dimension
            is the sequence dimension over which we want to aggregate of shape
            [seq_len, feat_dim]
          pooling_operation: either max pooling or average pooling
          only_non_zero_vectors: determines whether the aggregation is done over
            non-zero vectors only
        Returns:
          a matrix of shape [1, feat_dim]
        """
        shape = dense_sequence_features.shape
        if len(shape) != 2 or min(shape) == 0:
            raise ValueError(
                f"Expected a non-empty 2-dimensional matrix (where the first "
                f"dimension is the sequence dimension which we want to aggregate), "
                f"but found a matrix of shape {dense_sequence_features.shape}."
            )

        if only_non_zero_vectors:
            # take only non zeros feature vectors into account
            is_non_zero_vector = [f.any() for f in dense_sequence_features]
            dense_sequence_features = dense_sequence_features[is_non_zero_vector]

            # if features are all zero, then we must continue with zeros
            if not len(dense_sequence_features):
                dense_sequence_features = np.zeros([1, shape[-1]])

        if pooling_operation == MEAN_POOLING:
            return np.mean(dense_sequence_features, axis=0, keepdims=True)
        elif pooling_operation == MAX_POOLING:
            return np.max(dense_sequence_features, axis=0, keepdims=True)
        else:
            raise InvalidConfigException(
                f"Invalid pooling operation specified. Available operations are "
                f"'{MEAN_POOLING}' or '{MAX_POOLING}', but provided value is "
                f"'{pooling_operation}'."
            )
