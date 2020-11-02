from typing import Union, Text, Optional, List, Any, Tuple

import numpy as np
import scipy.sparse

from rasa.shared.nlu.constants import VALID_FEATURE_TYPES


class Features:
    """Stores the features produces by any featurizer."""

    def __init__(
        self,
        features: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_type: Text,
        attribute: Text,
        origin: Union[Text, List[Text]],
    ) -> None:
        self._validate_feature_type(feature_type)

        self.features = features
        self.type = feature_type
        self.origin = origin
        self.attribute = attribute

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

    def combine_with_features(self, additional_features: Optional["Features"]) -> None:
        """Combine the incoming features with this instance's features.

        Args:
            additional_features: additional features to add

        Returns:
            Combined features.
        """
        if additional_features is None:
            return

        if self.is_dense() and additional_features.is_dense():
            self._combine_dense_features(additional_features)
        elif self.is_sparse() and additional_features.is_sparse():
            self._combine_sparse_features(additional_features)
        else:
            raise ValueError("Cannot combine sparse and dense features.")

    def _combine_dense_features(self, additional_features: "Features") -> None:
        if self.features.ndim != additional_features.features.ndim:
            raise ValueError(
                f"Cannot combine dense features as sequence dimensions do not "
                f"match: {self.features.ndim} != {additional_features.features.ndim}."
            )

        self.features = np.concatenate(
            (self.features, additional_features.features), axis=-1
        )

    def _combine_sparse_features(self, additional_features: "Features") -> None:
        from scipy.sparse import hstack

        if self.features.shape[0] != additional_features.features.shape[0]:
            raise ValueError(
                f"Cannot combine sparse features as sequence dimensions do not "
                f"match: {self.features.shape[0]} != {additional_features.features.shape[0]}."
            )

        self.features = hstack([self.features, additional_features.features])

    def __key__(
        self,
    ) -> Tuple[
        Text, Text, Union[np.ndarray, scipy.sparse.spmatrix], Union[Text, List[Text]]
    ]:
        return (self.type, self.attribute, self.features, self.origin)

    def __hash__(self) -> int:
        return hash(self.__key__())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Features):
            return False

        return (
            other.type == self.type
            and other.attribute == self.attribute
            and other.features == self.features
        )
