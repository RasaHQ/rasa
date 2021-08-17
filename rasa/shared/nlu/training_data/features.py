from os import stat
from typing import Iterable, Union, Text, Optional, List, Any, Tuple, Set, Dict

import copy
import numpy as np
import scipy.sparse


class Features:
    """Stores the features produced by any featurizer."""

    def __init__(
        self,
        features: Union[np.ndarray, scipy.sparse.spmatrix],
        feature_type: Text,
        attribute: Text,
        origin: Union[Text, List[Text]],
    ) -> None:
        """Initializes the Features object.

        Args:
            features: The features.
            feature_type: Type of the feature, e.g. FEATURE_TYPE_SENTENCE.
            attribute: Message attribute, e.g. INTENT or TEXT.
            origin: Name of the component that created the features.
        """
        self.features = features
        self.type = feature_type
        self.origin = origin
        self.attribute = attribute

    def __repr__(self) -> Text:
        return (
            f"{self.__class__.__name__}("
            f"features={self.features}, "
            f"type={self.type}, "
            f"origin={self.origin}, "
            f"attribute={self.attribute})"
        )

    def __str__(self) -> Text:
        return (
            f"{self.__class__.__name__}("
            f"features.shape={self.features.shape}, "
            f"is_sparse={self.is_sparse()}, "
            f"type={self.type}, "
            f"origin={self.origin}, "
            f"attribute={self.attribute})"
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

    @staticmethod
    def filter(
        features_list: List["Features"],
        attributes: Optional[Iterable[Text]] = None,
        type: Optional[Text] = None,
        origin: Optional[List[Text]] = None,
        is_sparse: Optional[bool] = None,
    ) -> List["Features"]:
        """Filters the given list of features.

        Args:
          origin: If specified, this method will check that the exact order of origins
            matches the given list of origins. The reason for this is that if
            multiple origins are listed for a Feature, this means that this feature
            has been created by concatenating Features from the listed origins in
            that particular order.

        """
        filtered = features_list
        if attributes is not None:
            attributes = set(attributes)
            filtered = [f for f in filtered if f.attribute in attributes]
        if origin is not None:
            filtered = [
                f
                for f in filtered
                if (f.origin if not isinstance(f.origin, Text) else list([f.origin]))
                == origin
            ]
        if type is not None:
            filtered = [f for f in filtered if f.type == type]
        if is_sparse is not None:
            filtered = [f for f in filtered if f.is_sparse() == is_sparse]
        return filtered

    @staticmethod
    def extract(
        features_list: List["Features"], attributes: Optional[Iterable[Text]] = None,
    ) -> Dict[Text, List["Features"]]:
        if attributes is None:
            attributes = copy.copy(attributes)  # shallow copy on purpose
        # ensure all requested attributes are present in the output - regardless
        # of whether we find features later
        extracted = (
            dict()
            if attributes is None
            else {attribute: [] for attribute in attributes}
        )
        # extract features for all (requested) attributes
        for feat in features_list:
            if attributes is None or feat.attribute in attributes:
                extracted.setdefault(feat.attribute, []).append(feat)
        return extracted

    @staticmethod
    def combine(
        features_list: List["Features"],
        origin_of_combination: Optional[List[Text]] = None,
    ) -> "Features":
        """Combine features of the same type and level that describe the same attribute.

        If sequence features are to be combined, then they must have the same
        sequence dimension.

        Args:
          features: non-empty list of Features  of the same type and level that
            describe the same attribute
          origin_of_combination: origin information to be used for the resulting
            Feature
        Raises:
          `ValueError` will be raised
           - if the given list is empty
           - if there are inconsistencies in the given list of `Features`
           - if the origin information of a given feature is not listed in the given
             list of origins (i.e. `origin_of_combination`)
        """
        if len(features_list) == 0:
            raise ValueError("Expected a non-empty list of Features.")
        # sanity checks
        # (1) all origins must be mentioned and must not violate the given order
        minimal_origin = set(f.origin for f in features_list)
        if origin_of_combination is not None:
            difference = minimal_origin.difference(origin_of_combination)
            if difference:
                raise ValueError(
                    f"Expected given features to be from {origin_of_combination} only "
                    f"but found features from {difference}."
                )
            origin_of_combination_pruned = [
                origin for origin in origin_of_combination if origin in minimal_origin
            ]
            for idx, (origin, feat) in enumerate(
                zip(origin_of_combination_pruned, features_list)
            ):
                if feat.origin != origin:
                    raise ValueError(
                        f"Expected {origin} to be the origin of the {idx}-th feature "
                        f"(because of `origin_of_combination`) but found {feat.origin}."
                    )
        else:
            origin_of_combination = list(minimal_origin)
        # (2) certain attributes (is_sparse, type, attribute) must coincide
        for attribute in ["is_sparse", "type", "attribute"]:
            different_settings = set(getattr(f, attribute) for f in features_list)
            if attribute == "is_sparse":  # because this is a function atm
                different_settings = set(
                    is_sparse_func() for is_sparse_func in different_settings
                )
            if len(different_settings) > 1:
                raise ValueError(
                    f"Expected all Features to have the same {attribute} but found "
                    f" {different_settings}."
                )
        # (3) dimensions must match
        # Note: We shouldn't have to check sentence-level features here but it doesn't
        # hurt either.
        dimensions = set(f.features.shape[0] for f in features_list)
        if len(dimensions) > 1:
            raise ValueError(
                f"Expected all sequence dimensions to match but found {dimensions}."
            )
        # Combine the features
        arbitrary_feature = features_list[0]
        if not arbitrary_feature.is_sparse():
            features = np.concatenate((f.features for f in features_list), axis=-1)
        else:
            from scipy.sparse import hstack

            features = hstack([f.features for f in features_list])
        return Features(
            features=features,
            feature_type=arbitrary_feature.type,
            attribute=arbitrary_feature.attribute,
            origin=origin_of_combination,
        )

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
        """Returns a 4-tuple of defining properties.

        Returns:
            Tuple of type, attribute, features, and origin properties.
        """
        return (self.type, self.attribute, self.features, self.origin)

    def __eq__(self, other: Any) -> bool:
        """Tests if the `self` `Feature` equals to the `other`.

        Args:
            other: The other object.

        Returns:
            `True` when the other object is a `Feature` and has the same
            type, attribute, and feature tensors.
        """
        if not isinstance(other, Features):
            return False

        return (
            other.type == self.type
            and other.attribute == self.attribute
            and other.features == self.features
        )
