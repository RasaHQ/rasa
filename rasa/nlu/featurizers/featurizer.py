import numpy as np
import scipy.sparse
from typing import Text, Union, Optional, Dict, Any

from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.nlu.constants import VALID_FEATURE_TYPES
from rasa.nlu.components import Component
from rasa.utils.tensorflow.constants import MEAN_POOLING, MAX_POOLING


class Features:
    """Stores the features produces by any featurizer."""

    def __init__(
        self,
        features: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_type: Text,
        message_attribute: Text,
        origin: Text,
    ) -> None:
        self._validate_feature_type(feature_type)

        self.features = features
        self.type = feature_type
        self.origin = origin
        self.message_attribute = message_attribute

    @staticmethod
    def _validate_feature_type(feature_type: Text) -> None:
        if feature_type not in VALID_FEATURE_TYPES:
            raise ValueError(
                f"Invalid feature type '{feature_type}' used. Valid feature types are: "
                f"{VALID_FEATURE_TYPES}."
            )

    def is_sparse(self) -> bool:
        """Checks if features are sparse or not.

        Returns:
            True, if features are sparse, false otherwise.
        """
        return isinstance(self.features, scipy.sparse.spmatrix)

    def is_dense(self) -> bool:
        """Checks if features are dense or not.

        Returns:
            True, if features are dense, false otherwise.
        """
        return not self.is_sparse()

    def combine_with_features(
        self, additional_features: Optional[Union[np.ndarray, scipy.sparse.spmatrix]]
    ) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
        """Combine the incoming features with this instance's features.

        Args:
            additional_features: additional features to add

        Returns:
            Combined features.
        """
        if additional_features is None:
            return self.features

        if self.is_dense() and isinstance(additional_features, np.ndarray):
            return self._combine_dense_features(self.features, additional_features)

        if self.is_sparse() and isinstance(additional_features, scipy.sparse.spmatrix):
            return self._combine_sparse_features(self.features, additional_features)

        raise ValueError("Cannot combine sparse and dense features.")

    @staticmethod
    def _combine_dense_features(
        features: np.ndarray, additional_features: np.ndarray
    ) -> np.ndarray:
        if features.ndim != additional_features.ndim:
            raise ValueError(
                f"Cannot combine dense features as sequence dimensions do not "
                f"match: {features.ndim} != {additional_features.ndim}."
            )

        return np.concatenate((features, additional_features), axis=-1)

    @staticmethod
    def _combine_sparse_features(
        features: scipy.sparse.spmatrix, additional_features: scipy.sparse.spmatrix
    ) -> scipy.sparse.spmatrix:
        from scipy.sparse import hstack

        if features.shape[0] != additional_features.shape[0]:
            raise ValueError(
                f"Cannot combine sparse features as sequence dimensions do not "
                f"match: {features.shape[0]} != {additional_features.shape[0]}."
            )

        return hstack([features, additional_features])


class Featurizer(Component):
    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        if not component_config:
            component_config = {}

        # makes sure the alias name is set
        component_config.setdefault(FEATURIZER_CLASS_ALIAS, self.name)

        super().__init__(component_config)


class DenseFeaturizer(Featurizer):
    @staticmethod
    def _calculate_sentence_features(
        features: np.ndarray, pooling_operation: Text
    ) -> np.ndarray:
        # take only non zeros feature vectors into account
        non_zero_features = np.array([f for f in features if f.any()])

        # if features are all zero just return a vector with all zeros
        if non_zero_features.size == 0:
            return np.zeros([1, features.shape[-1]])

        if pooling_operation == MEAN_POOLING:
            return np.mean(non_zero_features, axis=0, keepdims=True)

        if pooling_operation == MAX_POOLING:
            return np.max(non_zero_features, axis=0, keepdims=True)

        raise ValueError(
            f"Invalid pooling operation specified. Available operations are "
            f"'{MEAN_POOLING}' or '{MAX_POOLING}', but provided value is "
            f"'{pooling_operation}'."
        )


class SparseFeaturizer(Featurizer):
    pass
