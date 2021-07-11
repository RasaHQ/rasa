from typing import Any, Optional, Tuple, Text, Dict, Set, List

import typing
import copy

import rasa.shared.utils.io
from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    RESPONSE,
    INTENT_RESPONSE_KEY,
    METADATA,
    METADATA_INTENT,
    METADATA_EXAMPLE,
    ENTITIES,
    ENTITY_ATTRIBUTE_START,
    ENTITY_ATTRIBUTE_END,
    RESPONSE_IDENTIFIER_DELIMITER,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
    ACTION_NAME,
)
from rasa.shared.constants import DIAGNOSTIC_DATA

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


class Message:
    def __init__(
        self,
        data: Optional[Dict[Text, Any]] = None,
        output_properties: Optional[Set] = None,
        time: Optional[int] = None,
        features: Optional[List["Features"]] = None,
        **kwargs: Any,
    ) -> None:
        self.time = time
        self.data = data.copy() if data else {}
        self.features = features if features else []

        self.data.update(**kwargs)

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()
        self.output_properties.add(TEXT)

    def add_features(self, features: Optional["Features"]) -> None:
        if features is not None:
            self.features.append(features)

    def add_diagnostic_data(self, origin: Text, data: Dict[Text, Any]) -> None:
        """Adds diagnostic data from the `origin` component.

        Args:
            origin: Name of the component that created the data.
            data: The diagnostic data.
        """
        if origin in self.get(DIAGNOSTIC_DATA, {}):
            rasa.shared.utils.io.raise_warning(
                f"Please make sure every pipeline component has a distinct name. "
                f"The name '{origin}' appears at least twice and diagnostic "
                f"data will be overwritten."
            )
        self.data.setdefault(DIAGNOSTIC_DATA, {})
        self.data[DIAGNOSTIC_DATA][origin] = data

    def set(self, prop: Text, info: Any, add_to_output: bool = False) -> None:
        """Sets the message's property to the given value.

        Args:
            prop: Name of the property to be set.
            info: Value to be assigned to that property.
            add_to_output: Decides whether to add `prop` to the `output_properties`.
        """
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        return self.data.get(prop, default)

    def as_dict_nlu(self) -> dict:
        """Get dict representation of message as it would appear in training data"""

        d = self.as_dict()
        if d.get(INTENT, None):
            d[INTENT] = self.get_full_intent()
        d.pop(RESPONSE, None)
        d.pop(INTENT_RESPONSE_KEY, None)
        return d

    def as_dict(self, only_output_properties: bool = False) -> Dict:
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
        return {key: value for key, value in d.items() if value is not None}

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Message):
            return False
        else:
            return other.fingerprint() == self.fingerprint()

    def __hash__(self) -> int:
        """Calculate a hash for the message.

        Returns:
            Hash of the message.
        """
        return int(self.fingerprint(), 16)

    def fingerprint(self) -> Text:
        """Calculate a string fingerprint for the message.

        Returns:
            Fingerprint of the message.
        """
        return rasa.shared.utils.io.deep_container_fingerprint(self.data)

    @classmethod
    def build(
        cls,
        text: Text,
        intent: Optional[Text] = None,
        entities: Optional[List[Dict[Text, Any]]] = None,
        intent_metadata: Optional[Any] = None,
        example_metadata: Optional[Any] = None,
        **kwargs: Any,
    ) -> "Message":
        """Builds a Message from `UserUttered` data.

        Args:
            text: text of a user's utterance
            intent: an intent of the user utterance
            entities: entities in the user's utterance
            intent_metadata: optional metadata for the intent
            example_metadata: optional metadata for the intent example

        Returns:
            Message
        """
        data: Dict[Text, Any] = {TEXT: text}
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
        if intent_metadata is not None:
            data[METADATA] = {METADATA_INTENT: intent_metadata}
        if example_metadata is not None:
            data.setdefault(METADATA, {})[METADATA_EXAMPLE] = example_metadata

        return cls(data, **kwargs)

    def get_full_intent(self) -> Text:
        """Get intent as it appears in training data"""

        return (
            self.get(INTENT_RESPONSE_KEY)
            if self.get(INTENT_RESPONSE_KEY)
            else self.get(INTENT)
        )

    def get_combined_intent_response_key(self) -> Text:
        """Get intent as it appears in training data."""
        rasa.shared.utils.io.raise_warning(
            "`get_combined_intent_response_key` is deprecated and "
            "will be removed in Rasa 3.0.0. "
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
    ) -> Tuple[Optional["Features"], Optional["Features"]]:
        """Gets all sparse features for the attribute given the list of featurizers.

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

        sequence_features = self._combine_features(sequence_features, featurizers)
        sentence_features = self._combine_features(sentence_features, featurizers)

        return sequence_features, sentence_features

    def get_sparse_feature_sizes(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> Dict[Text, List[int]]:
        """Gets sparse feature sizes for the attribute given the list of featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: message attribute
            featurizers: names of featurizers to consider

        Returns:
            Sparse feature sizes.
        """
        if featurizers is None:
            featurizers = []

        sequence_features, sentence_features = self._filter_sparse_features(
            attribute, featurizers
        )
        sequence_sizes = [f.features.shape[1] for f in sequence_features]
        sentence_sizes = [f.features.shape[1] for f in sentence_features]

        return {
            FEATURE_TYPE_SEQUENCE: sequence_sizes,
            FEATURE_TYPE_SENTENCE: sentence_sizes,
        }

    def get_dense_features(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> Tuple[Optional["Features"], Optional["Features"]]:
        """Gets all dense features for the attribute given the list of featurizers.

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

        sequence_features = self._combine_features(sequence_features, featurizers)
        sentence_features = self._combine_features(sentence_features, featurizers)

        return sequence_features, sentence_features

    def get_all_features(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> List["Features"]:
        """Gets all features for the attribute given the list of featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: message attribute
            featurizers: names of featurizers to consider

        Returns:
            Features.
        """
        sparse_features = self.get_sparse_features(attribute, featurizers)
        dense_features = self.get_dense_features(attribute, featurizers)

        return [f for f in sparse_features + dense_features if f is not None]

    def features_present(
        self, attribute: Text, featurizers: Optional[List[Text]] = None
    ) -> bool:
        """Checks if there are any features present for the attribute and featurizers.

        If no featurizers are provided, all available features will be considered.

        Args:
            attribute: Message attribute.
            featurizers: Names of featurizers to consider.

        Returns:
            ``True``, if features are present, ``False`` otherwise.
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
            if f.attribute == attribute
            and f.is_dense()
            and f.type == FEATURE_TYPE_SENTENCE
            and (f.origin in featurizers or not featurizers)
        ]
        sequence_features = [
            f
            for f in self.features
            if f.attribute == attribute
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
            if f.attribute == attribute
            and f.is_sparse()
            and f.type == FEATURE_TYPE_SENTENCE
            and (f.origin in featurizers or not featurizers)
        ]
        sequence_features = [
            f
            for f in self.features
            if f.attribute == attribute
            and f.is_sparse()
            and f.type == FEATURE_TYPE_SEQUENCE
            and (f.origin in featurizers or not featurizers)
        ]

        return sequence_features, sentence_features

    @staticmethod
    def _combine_features(
        features: List["Features"], featurizers: Optional[List[Text]] = None
    ) -> Optional["Features"]:
        combined_features = None

        for f in features:
            if combined_features is None:
                combined_features = copy.deepcopy(f)
                combined_features.origin = featurizers
            else:
                combined_features.combine_with_features(f)

        return combined_features

    def is_core_or_domain_message(self) -> bool:
        """Checks whether the message is a core message or from the domain.

        E.g. a core message is created from a story or a domain action,
        not from the NLU data.

        Returns:
            True, if message is a core or domain message, false otherwise.
        """
        return bool(
            self.data.get(ACTION_NAME)
            or self.data.get(ACTION_TEXT)
            or (
                (self.data.get(INTENT) or self.data.get(RESPONSE))
                and not self.data.get(TEXT)
            )
            or (
                self.data.get(TEXT)
                and not (self.data.get(INTENT) or self.data.get(RESPONSE))
            )
        )

    def is_e2e_message(self) -> bool:
        """Checks whether the message came from an e2e story.

        Returns:
            `True`, if message is a from an e2e story, `False` otherwise.
        """
        return bool(
            (self.get(ACTION_TEXT) and not self.get(ACTION_NAME))
            or (self.get(TEXT) and not self.get(INTENT))
        )

    def find_overlapping_entities(
        self,
    ) -> List[Tuple[Dict[Text, Any], Dict[Text, Any]]]:
        """Finds any overlapping entity annotations."""
        entities = self.get("entities", [])[:]
        entities_with_location = [
            e
            for e in entities
            if (ENTITY_ATTRIBUTE_START in e.keys() and ENTITY_ATTRIBUTE_END in e.keys())
        ]
        entities_with_location.sort(key=lambda e: e[ENTITY_ATTRIBUTE_START])
        overlapping_pairs: List[Tuple[Dict[Text, Any], Dict[Text, Any]]] = []
        for i, entity in enumerate(entities_with_location):
            for other_entity in entities_with_location[i + 1 :]:
                if other_entity[ENTITY_ATTRIBUTE_START] < entity[ENTITY_ATTRIBUTE_END]:
                    overlapping_pairs.append((entity, other_entity))
                else:
                    break
        return overlapping_pairs
