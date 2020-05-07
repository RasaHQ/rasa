from typing import Any, Optional, Tuple, Text, Dict, Set, List, Union

import numpy as np
import scipy.sparse

from rasa.nlu.constants import (
    ENTITIES,
    INTENT,
    RESPONSE,
    RESPONSE_KEY_ATTRIBUTE,
    TEXT,
    RESPONSE_IDENTIFIER_DELIMITER,
    SEQUENCE,
    SENTENCE,
)
from rasa.nlu.utils import ordered


class Message:
    def __init__(
        self,
        text: Text,
        data: Optional[Dict[Text, Any]] = None,
        output_properties: Optional[Set] = None,
        time: Optional[Text] = None,
        features: Optional[List["Features"]] = None,
    ) -> None:
        self.text = text
        self.time = time
        self.data = data if data else {}
        self.features = features if features else []

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def add_features(self, features: "Features") -> None:
        self.features.append(features)

    def set(self, prop, info, add_to_output=False) -> None:
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
            d[INTENT] = self.get_combined_intent_response_key()
        d.pop(RESPONSE_KEY_ATTRIBUTE, None)
        d.pop(RESPONSE, None)
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

        # Filter all keys with None value. These could have come while building the Message object in markdown format
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
    def build(cls, text, intent=None, entities=None) -> "Message":
        data = {}
        if intent:
            split_intent, response_key = cls.separate_intent_response_key(intent)
            data[INTENT] = split_intent
            if response_key:
                data[RESPONSE_KEY_ATTRIBUTE] = response_key
        if entities:
            data[ENTITIES] = entities
        return cls(text, data)

    def get_combined_intent_response_key(self) -> Text:
        """Get intent as it appears in training data"""

        intent = self.get(INTENT)
        response_key = self.get(RESPONSE_KEY_ATTRIBUTE)
        response_key_suffix = (
            f"{RESPONSE_IDENTIFIER_DELIMITER}{response_key}" if response_key else ""
        )
        return f"{intent}{response_key_suffix}"

    @staticmethod
    def separate_intent_response_key(original_intent) -> Optional[Tuple[Any, Any]]:

        split_title = original_intent.split(RESPONSE_IDENTIFIER_DELIMITER)
        if len(split_title) == 2:
            return split_title[0], split_title[1]
        elif len(split_title) == 1:
            return split_title[0], None

    def _filter_features(
        self,
        attribute: Text,
        sequence_featurizers: List[Text],
        sentence_featurizers: List[Text],
        sparse: bool,
    ) -> Tuple[Optional[List["Features"]], Optional[List["Features"]]]:
        if sparse:
            features = [
                f
                for f in self.features
                if f.message_attribute == attribute and f.is_sparse()
            ]
        else:
            features = [
                f
                for f in self.features
                if f.message_attribute == attribute and f.is_dense()
            ]

        if not features:
            return None, None

        sequence_features = [
            f
            for f in features
            if f.type == SEQUENCE
            and (f.origin in sequence_featurizers or not sequence_featurizers)
        ]
        sentence_features = [
            f
            for f in features
            if f.type == SENTENCE
            and (f.origin in sentence_featurizers or not sentence_featurizers)
        ]

        return sequence_features, sentence_features

    def get_sparse_features(
        self, attribute: Text, sequence_featurizers: List, sentence_featurizers: List
    ) -> Tuple[
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
    ]:

        sequence_features, sentence_features = self._filter_features(
            attribute, sequence_featurizers, sentence_featurizers, sparse=True
        )

        if not sequence_features and not sentence_features:
            return None, None

        return self._combine_features(sequence_features, sentence_features)

    @staticmethod
    def _combine_features(
        sequence_features: List["Features"], sentence_features: List["Features"]
    ) -> Tuple[
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
    ]:
        from rasa.nlu.featurizers.featurizer import Features

        combined_sequence_features = None
        for f in sequence_features:
            combined_sequence_features = Features.combine_features(
                combined_sequence_features, f
            )
        combined_sentence_features = None
        for f in sentence_features:
            combined_sentence_features = Features.combine_features(
                combined_sentence_features, f
            )
        return combined_sequence_features, combined_sentence_features

    def get_dense_features(
        self, attribute: Text, sequence_featurizers: List, sentence_featurizers: List
    ) -> Tuple[
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
        Optional[List[Union[np.ndarray, scipy.sparse.spmatrix]]],
    ]:
        sequence_features, sentence_features = self._filter_features(
            attribute, sequence_featurizers, sentence_featurizers, sparse=False
        )

        if not sequence_features and not sentence_features:
            return None, None

        return self._combine_features(sequence_features, sentence_features)
