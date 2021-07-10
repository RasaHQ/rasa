"""
NOTE:
- it would be great if we could turn the state featurizer into a combination of
  state item featurizers (cf. target featurizers)
-
"""

from abc import abstractmethod
import logging
import numpy as np
import scipy.sparse
from typing import Generic, List, Optional, Dict, Text, Any, Iterable

from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.shared.nlu.training_data.features import Features
from rasa.shared import io as io_utils
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.core.domain import Domain
from rasa.shared.core import state as state_utils
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.shared.nlu.constants import (
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ACTION_NAME,
    FEATURE_TYPE_SEQUENCE,
    INTENT,
    ENTITY_TAGS,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    USER,
    SLOTS,
)
from rasa.utils.tensorflow import model_data_utils
from rasa.nlu.utils import bilou_utils

logger = logging.getLogger(__name__)


def item2index_mapping(unique_items: Iterable[Text]) -> Dict[Text, int]:
    """Computes a mapping that maps each given label to a unique int index.

    The computed mapping is consistent across sessions since the unique_items are
    sorted before the mappin is computed.

    TODO: move this somewhere else..

    Raises:
      ValueError if the given labels are not unique
    """
    if len(set(unique_items)) != len(unique_items):
        raise ValueError("The given values are not unique.")
    # sorted to assure we get the same order with each session (just in case this
    # list was generated via from some set/dict/...)
    return {
        feature_state: idx for idx, feature_state in enumerate(sorted(unique_items))
    }


def convert_sparse_sequence_to_sentence_features(
    features: List["Features"],
) -> List["Features"]:
    """Grabs all sparse sequence features and turns them into sparse sentence features.

    That is, this function filters all sparse sequence features and then
    turns each of these sparse sequence features into sparse sentence feature
    by summing over their first axis, respectively.

    TODO: move this somewhere else
    TODO: extend features to obtain meaningful aggregations

    Returns:
      a list with as many sparse sentenc features as there are sparse sequence features
      in the given list
    """
    # TODO: add functionality to `Features` to obtain a meaningful conversion from
    # sequence to sentence features depending on it's "origin" (requires rework of
    # Features class to work properly because the origin attribute is really a
    # tag defined by the user...)
    return [
        Features(
            scipy.sparse.coo_matrix(feature.features.sum(0)),
            FEATURE_TYPE_SENTENCE,
            feature.attribute,
            feature.origin,
        )
        for feature in features
        if (feature.is_sparse and feature.type == FEATURE_TYPE_SEQUENCE)
    ]


class SubStateFeaturizer:
    @abstractmethod
    def setup_domain(self, domain: Domain) -> None:
        # TODO: that name is debatable...
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    def raise_if_ready_is(self, should_be_ready: bool) -> None:
        if not self.is_ready() and should_be_ready:
            raise RuntimeError(
                f"Expected {self.__class__.__name__} has been `setup()`."
            )
        elif self.is_ready() and not should_be_ready:
            raise RuntimeError(
                f"Expected {self.__class__.__name__} had not been `setup()` before."
            )

    @abstractmethod
    def featurize(
        self, sub_state: SubState, attributes: Optional[List[Text]] = None
    ) -> Dict[Text, List[Features]]:
        pass


class FallbackSubStateFeaturizer(SubStateFeaturizer):
    # TODO: version that converts to indices, from which fallback featurizer inherits
    # TODO: in the old versoin, padding was done *here* -> need to introduce some
    # further glue component to batch the indices.... or just store them in sparse
    # matrices just like everythin else here... -> check why indices
    def __init__(
        self, item2index_mappings: Optional[Dict[Text, Dict[Text, Any]]] = None,
    ) -> None:
        self.item2index_mappings = item2index_mappings

    def is_ready(self) -> bool:
        return self.item2index_mappings is not None

    def setup_domain(self, domain: Domain) -> None:
        self.raise_if_ready_is(True)
        self.item2index_mappings = {
            key: item2index_mapping(items)
            for key, items in [
                (INTENT, domain.intents),
                (ACTION_NAME, domain.action_names_or_texts),
                (ENTITIES, domain.entity_states),
                (SLOTS, domain.slot_states),
                (ACTIVE_LOOP, domain.form_names),
            ]
        }
        assert set(self.attributes) <= set(self.item2index_mappings.keys())

    def _encoding(self, sub_state: SubState, attribute: Text):
        """Turns the given mapping from a categorical variable name to a value.

        Note that the space of possible cateorical variables is determined by the
        domain that was used to setup this featurizer.

        # NOTE: this was `_state_features_for_attribute`
        """
        # FIXME: Create dataclasses for states, substates, messages,...
        #   Then, e.g. the following would be type-safe.

        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}  # type: ignore[dict-item]
        elif attribute == ENTITIES:
            # TODO/FIXME: no bilou tagging handled here, in contrast
            # featurize_entities
            return {entity: 1 for entity in sub_state.get(ENTITIES, [])}
        elif attribute == ACTIVE_LOOP:
            return {sub_state["name"]: 1}  # type: ignore[dict-item]
        elif attribute == SLOTS:
            return {
                f"{slot_name}_{i}": value
                for slot_name, slot_as_feature in sub_state.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self.item2index_mappings.keys()}'."
            )

    def _featurize(self, sub_state: SubState, attribute: Text) -> Features:
        """Creates a multi-hot encoding like feature for the given attribute.

        Leverages the item2index mappings computed during `prepare_training`.
        """
        encoding: Dict[Text, int] = self._encoding(sub_state, attribute)

        # FIXME: always sparse? (dense on demand)
        features = np.zeros(len(self.item2index_mapping), np.float32)
        for feature_name, value in encoding.items():
            if feature_name in self.item2index_mapping:
                features[self.item2index_mappings[feature_name]] = value
        features = np.expand_dims(features, 0)

        if self.sparse:
            features = scipy.sparse.coo_matrix(features)

        return Features(
            features,
            FEATURE_TYPE_SENTENCE,
            self.attribute,
            origin=self.__class__.__name__,
        )

    def featurize(
        self, sub_state: SubState, attributes: List[Text]
    ) -> Dict[Text, List[Features]]:
        self.raise_if_ready_is(False)
        return {
            attribute: self._featurize(sub_state, attribute) for attribute in attributes
        }


class SubStateFeaturizerUsingInterpreter(SubStateFeaturizer):
    def __init__(
        self,
        keep_only_sparse_seq_converted_to_sent: Optional[List[Text]],
        use_fallback_interpreter: Optional[List[Text]],
        use_regex_interpreter: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.use_regex_interpreter = use_regex_interpreter
        self.keep_only_sparse_seq_converted_to_sent = (
            keep_only_sparse_seq_converted_to_sent
        )
        if use_fallback_interpreter:
            self.fallback_featurizer = FallbackSubStateFeaturizer(
                attributes=use_fallback_interpreter
            )

    def setup_interpreter(
        self, interpreter: Optional[NaturalLanguageInterpreter],
    ) -> None:
        # do NOT raise an exception here
        if self.use_regex_interpreter is not None:
            self.use_regex_interpreter = (
                isinstance(interpreter, RegexInterpreter) if interpreter else False
            )
        self.interpreter = interpreter

    def is_ready(self) -> bool:
        return self.interpreter is not None

    def featurize(
        self, sub_state: SubState
    ) -> Dict[Text, List[Features]]:  # FIXME: attributes

        self.raise_if_ready_is(False)

        # (1) featurize the message/sub_state...
        message = Message(data=sub_state)
        parsed_message = self.interpreter.featurize_message(message)
        # TODO: is it clear that this works? if not, it should be made explicit
        # somewhere somehow .... -> SubState and Message dataclass things....

        # (2) gather all features from the featurized message -- except the ones
        # we are told to `exclude`
        attribute_to_features: Dict[Text, List[Features]] = dict()
        for attribute in self.attributes:
            all_features = parsed_message.get_sparse_features(
                attribute
            ) + parsed_message.get_dense_features(attribute)

            for features in all_features:
                if features is not None:
                    attribute_to_features[attribute].append(features)

        # (3) for the attributes in `keep_only_sparse_seq_converted_to_sent`
        # create a sentence feature from the sparse sequence features
        # and forget all other features (iff such features exist, otherwise
        # keep the existing sentence feature)
        attribute_list = self.keep_only_sparse_seq_converted_to_sent or []
        for attribute in attribute_list:
            if attribute_to_features.get(attribute):
                converted_features = convert_sparse_sequence_to_sentence_features(
                    attribute_to_features[attribute]
                )
                if converted_features:
                    attribute_to_features[attribute] = converted_features

        # (4) for all attributes listed in `use_fallback_interpreter`, *for which
        # no features have been extracted via the interpreter until now*, use the
        # `featurize_attribute_via_fallback_interpreter` method to create features
        # from scratch
        attribute_list = self.use_fallback_interpreter or []
        for attribute in attribute_list:
            if attribute not in attribute_to_features:
                attribute_to_features[attribute] = self.fallback_featurizer.featurize(
                    sub_state, attribute, sparse=self.sparse,
                )
        return attribute_to_features


class ActionAttributeFeaturizer(SubStateFeaturizerUsingInterpreter):
    def __init__(
        self, use_regex_interpreter: Optional[bool] = None,
    ):

        super().__init__(
            # attributes=[ACTION_NAME, ACTION_TEXT],
            keep_only_sparse_seq_converted_to_sent=None,
            use_fallback_interpreter=None,
            use_regex_interpreter=use_regex_interpreter,
        )

    # TODO: check whether attribute == action in featurize (super featurizes all)

    def featurize_action(self, action: Text) -> Dict[Text, List["Features"]]:
        """


        """
        self.raise_if_ready_is(False)

        sub_state = state_utils.create_substate_from_action(
            action, as_text=(action in self.action_texts)
        )
        return self.featurize(sub_state)


class EntityAttributeFeaturizer(SubStateFeaturizer):
    def __init__(
        self, bilou_tagging: bool, encoding_spec: Optional[EntityTagSpec] = None
    ) -> None:
        """

        """
        super().__init__(attributes=...)  # TODO
        self.bilou_tagging = bilou_tagging  # this should be known in advance?
        self.encoding_spec = encoding_spec

    def setup_domain(self, domain: Domain):
        entities = sorted(domain.entity_states)
        if self.bilou_tagging:
            tag_id_index_mapping = {
                f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
                for tag, idx_1 in enumerate(entities)
                for idx_2, prefix in enumerate(BILOU_PREFIXES)
            }
        else:
            tag_id_index_mapping = {
                tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
                for tag, idx in enumerate(entities)
            }

        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_index_mapping[NO_ENTITY_TAG] = 0

        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        #  we return a list to anticipate that
        self.encoding_spec = EntityTagSpec(
            tag_name=ENTITY_ATTRIBUTE_TYPE,
            tags_to_ids=tag_id_index_mapping,
            ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
            num_tags=len(tag_id_index_mapping),
        )

    def featurize(
        self, sub_state: SubState, attributes: List[Text]
    ) -> Dict[Text, List[Features]]:
        self.raise_if_ready_is(False)
        assert attributes == [
            ENTITY_TAGS
        ]  # TODO: supported attributes def in __init__?
        self.raise_if_ready_is(False)

        message = Message(sub_state)
        message = self.interpreter.featurize_message(message)

        if not message:
            return {}

        if self.bilou_tagging:
            bilou_utils.apply_bilou_schema_to_message(message)

        return {
            ENTITY_TAGS: [
                model_data_utils.get_tag_ids(
                    message, self.encoding_specs, self.bilou_tagging
                )
            ]
        }
