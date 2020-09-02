from typing import Any, Optional, Tuple, Text, Dict, Set, List, Union

import numpy as np
import scipy.sparse
import typing

from rasa.exceptions import RasaException
from rasa.nlu.constants import (
    ENTITIES,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    TEXT,
    RESPONSE_IDENTIFIER_DELIMITER,
    FEATURE_TYPE_SEQUENCE,
    FEATURE_TYPE_SENTENCE,
)
from rasa.nlu.utils import ordered
from rasa.utils import common as common_utils

if typing.TYPE_CHECKING:
    from rasa.nlu.featurizers.featurizer import Features


class Message:
    def __init__(
        self,
        text: Text,
        data: Optional[Dict[Text, Any]] = None,
        output_properties: Optional[Set] = None,
        time: Optional[Text] = None,
        features: Optional[List["Features"]] = None,
        **kwargs,
    ) -> None:
        self.text = text
        self.time = time
        self.data = data if data else {}
        self.features = features if features else []

        self.data.update(**kwargs)

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def add_features(self, features: Optional["Features"]) -> None:
        if features is not None:
            self.features.append(features)

    def set(self, prop, info, add_to_output=False) -> None:
        if prop == TEXT:
            self.text = info
        else:
            self.data[prop] = info
            if add_to_output:
                self.output_properties.add(prop)

    def get(self, prop, default=None) -> Any:
        if prop == TEXT:
            return self.text
        return self.data.get(prop, default)

    def as_dict_nlu(self) -> dict:
        """Get dict representation of message as it would appear in training data"""

        d = self.as_dict()
        if d.get(INTENT, None):
            d[INTENT] = self.get_full_intent()
        d.pop(RESPONSE, None)
        d.pop(INTENT_RESPONSE_KEY, None)
        return d

    def as_dict(self, only_output_properties=False) -> dict:
        if only_output_properties:
            d = {
                key: value
                for key, value in self.data.items()
                if key in self.output_properties
            }
        else:
            d = self.data

        # Filter all keys with None value. These could have come while building the
        # Message object in markdown format
        d = {key: value for key, value in d.items() if value is not None}

        return dict(d, text=self.text)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Message):
            return False
        else:
            return (other.text, ordered(other.data)) == (self.text, ordered(self.data))

    def __hash__(self) -> int:
        return hash((self.text, str(ordered(self.data))))

    @classmethod
    def build(
        cls,
        text: Text,
        intent: Optional[Text] = None,
        entities: List[Dict[Text, Any]] = None,
        **kwargs,
    ) -> "Message":
        data = {}
        if intent:
            split_intent, response_key = cls.separate_intent_response_key(intent)
            if split_intent:
                data[INTENT] = split_intent
            if response_key:
                # intent label can be of the form - {intent}/{response_key},
                # so store the full intent label in intent_response_key
                data[INTENT_RESPONSE_KEY] = intent
        if entities:
            data[ENTITIES] = entities
        return cls(text, data, **kwargs)

    def get_full_intent(self) -> Text:
        """Get intent as it appears in training data"""

        return (
            self.get(INTENT_RESPONSE_KEY)
            if self.get(INTENT_RESPONSE_KEY)
            else self.get(INTENT)
        )

    def get_combined_intent_response_key(self) -> Text:
        """Get intent as it appears in training data"""

        common_utils.raise_warning(
            "`get_combined_intent_response_key` is deprecated and "
            "will be removed in future versions. "
            "Please use `get_full_intent` instead.",
            category=DeprecationWarning,
        )
        return self.get_full_intent()

    @staticmethod
    def separate_intent_response_key(
        original_intent: Text,
    ) -> Tuple[Text, Optional[Text]]:

        split_title = original_intent.split(RESPONSE_IDENTIFIER_DELIMITER)
        if len(split_title) == 2:
            return split_title[0], split_title[1]
        elif len(split_title) == 1:
            return split_title[0], None

        raise RasaException(
            f"Intent name '{original_intent}' is invalid, "
            f"it cannot contain more than one '{RESPONSE_IDENTIFIER_DELIMITER}'."
        )

    def get_sparse_features(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[scipy.sparse.spmatrix]]:
        """Get all sparse features for the given attribute that are coming from the
        given list of featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: message attribute
            featurizers: names of featurizers to consider

        Returns:
            Sparse features.
        """
        if featurizers is None:
            featurizers = []

        sequence_features, sentence_features = self._filter_sparse_features(
            attribute, featurizers
        )

        sequence_features = self._combine_features(sequence_features)
        sentence_features = self._combine_features(sentence_features)

        return sequence_features, sentence_features

    def get_dense_features(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get all dense features for the given attribute that are coming from the given
        list of featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: message attribute
            featurizers: names of featurizers to consider

        Returns:
            Dense features.
        """
        if featurizers is None:
            featurizers = []

        sequence_features, sentence_features = self._filter_dense_features(
            attribute, featurizers
        )

        sequence_features = self._combine_features(sequence_features)
        sentence_features = self._combine_features(sentence_features)

        return sequence_features, sentence_features

    def features_present(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> bool:
        """Check if there are any features present for the given attribute and
        featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: message attribute
            featurizers: names of featurizers to consider

        Returns:
            ``True``, if features are present, ``False`` otherwise
        """
        if featurizers is None:
            featurizers = []

        (
            sequence_sparse_features,
            sentence_sparse_features,
        ) = self._filter_sparse_features(attribute, featurizers)
        sequence_dense_features, sentence_dense_features = self._filter_dense_features(
            attribute, featurizers
        )

        return (
            len(sequence_sparse_features) > 0
            or len(sentence_sparse_features) > 0
            or len(sequence_dense_features) > 0
            or len(sentence_dense_features) > 0
        )

    def _filter_dense_features(
        self, attribute: Text, featurizers: List[Text]
    ) -> Tuple[List["Features"], List["Features"]]:
        sentence_features = [
            f
            for f in self.features
            if f.message_attribute == attribute
            and f.is_dense()
            and f.type == FEATURE_TYPE_SENTENCE
            and (f.origin in featurizers or not featurizers)
        ]
        sequence_features = [
            f
            for f in self.features
            if f.message_attribute == attribute
            and f.is_dense()
            and f.type == FEATURE_TYPE_SEQUENCE
            and (f.origin in featurizers or not featurizers)
        ]
        return sequence_features, sentence_features

    def _filter_sparse_features(
        self, attribute: Text, featurizers: List[Text]
    ) -> Tuple[List["Features"], List["Features"]]:
        sentence_features = [
            f
            for f in self.features
            if f.message_attribute == attribute
            and f.is_sparse()
            and f.type == FEATURE_TYPE_SENTENCE
            and (f.origin in featurizers or not featurizers)
        ]
        sequence_features = [
            f
            for f in self.features
            if f.message_attribute == attribute
            and f.is_sparse()
            and f.type == FEATURE_TYPE_SEQUENCE
            and (f.origin in featurizers or not featurizers)
        ]

        return sequence_features, sentence_features

    @staticmethod
    def _combine_features(
        features: List["Features"],
    ) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
        combined_features = None

        for f in features:
            combined_features = f.combine_with_features(combined_features)

        return combined_features
