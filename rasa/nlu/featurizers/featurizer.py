import numpy as np
import scipy.sparse
from typing import Any, Text, Union, Optional

from rasa.nlu.training_data import Message
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TEXT,
    VALID_FEATURE_TYPES,
)
from rasa.utils.tensorflow.constants import MEAN_POOLING, MAX_POOLING


def sequence_to_sentence_features(
    features: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
    """Extract the CLS token vector as sentence features.

    Features is a sequence. The last token is the CLS token. The feature vector of
    this token contains the sentence features."""

    if features is None:
        return None

    if isinstance(features, scipy.sparse.spmatrix):
        return scipy.sparse.coo_matrix(features.tocsr()[-1])

    return np.expand_dims(features[-1], axis=0)


class Features:
    def __init__(
        self,
        features: Union[np.ndarray, scipy.sparse.spmatrix],
        type: Text,
        message_attribute: Text,
        origin: Text,
    ):
        self.validate_type(type)

        self.features = features
        self.type = type
        self.origin = origin
        self.message_attribute = message_attribute

    def validate_type(self, type: Text):
        if type not in VALID_FEATURE_TYPES:
            raise ValueError(
                f"Invalid feature type '{type}' used. Valid feature types are: {VALID_FEATURE_TYPES}."
            )

    def is_sparse(self):
        return isinstance(self.features, scipy.sparse.spmatrix)

    def is_dense(self):
        return not self.is_sparse()

    @staticmethod
    def combine_features(
        features: Optional[Union[np.ndarray, scipy.sparse.spmatrix]],
        additional_features: "Features",
    ) -> Any:
        if features is None:
            return additional_features.features

        if additional_features.is_dense():
            return np.concatenate((features, additional_features.features), axis=-1)

        from scipy.sparse import hstack

        return hstack([features, additional_features.features])


class Featurizer(Component):
    pass


class DenseFeaturizer(Featurizer):
    @staticmethod
    def _combine_with_existing_dense_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = DENSE_FEATURE_NAMES[TEXT],
    ) -> Any:
        if message.get(feature_name) is not None:

            if len(message.get(feature_name)) != len(additional_features):
                raise ValueError(
                    f"Cannot concatenate dense features as sequence dimension does not "
                    f"match: {len(message.get(feature_name))} != "
                    f"{len(additional_features)}. Message: '{message.text}'."
                )

            return np.concatenate(
                (message.get(feature_name), additional_features), axis=-1
            )
        else:
            return additional_features

    @staticmethod
    def _calculate_cls_vector(
        features: np.ndarray, pooling_operation: Text
    ) -> np.ndarray:
        # take only non zeros feature vectors into account
        non_zero_features = np.array([f for f in features if f.any()])

        # if features are all zero just return a vector with all zeros
        if non_zero_features.size == 0:
            return np.zeros([1, features.shape[-1]])

        if pooling_operation == MEAN_POOLING:
            return np.mean(non_zero_features, axis=0, keepdims=True)
        elif pooling_operation == MAX_POOLING:
            return np.max(non_zero_features, axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Invalid pooling operation specified. Available operations are "
                f"'{MEAN_POOLING}' or '{MAX_POOLING}', but provided value is "
                f"'{pooling_operation}'."
            )


class SparseFeaturizer(Featurizer):
    @staticmethod
    def _combine_with_existing_sparse_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = SPARSE_FEATURE_NAMES[TEXT],
    ) -> Any:
        if additional_features is None:
            return

        if message.get(feature_name) is not None:
            from scipy.sparse import hstack

            if message.get(feature_name).shape[0] != additional_features.shape[0]:
                raise ValueError(
                    f"Cannot concatenate sparse features as sequence dimension does not "
                    f"match: {message.get(feature_name).shape[0]} != "
                    f"{additional_features.shape[0]}. Message: '{message.text}'."
                )
            return hstack([message.get(feature_name), additional_features])
        else:
            return additional_features
