from abc import abstractmethod
import logging
import numpy as np
import scipy.sparse
from typing import List, Optional, Dict, Text, Any, Iterable

from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.core.domain import Message, Domain
from rasa.shared.core.domain import Domain
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
    ACTIVE_LOOP,
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


def from_given_choose_supported_and_remove_excluded(
    given: Iterable[Text],
    supported: Optional[Iterable[Text]] = None,
    excluded: Optional[Iterable[Text]] = None,
) -> List[Text]:
    result = set(given)
    if supported:
        result = result.intersection(supported)
    if excluded:
        result = result.difference(excluded)
    result = sorted(result)
    return result


class SetupMixin:
    @abstractmethod
    def is_setup(self) -> bool:
        pass

    def raise_if_setup_is(self, should_be_setup: bool) -> None:
        if not self.is_setup() and should_be_setup:
            raise RuntimeError(f"Expected {self.__class__.__name__} to be configured.")
        elif self.is_setup() and not should_be_setup:
            raise RuntimeError(
                f"Expected {self.__class__.__name__} to not be configured yet."
            )


class MessageDataFeaturizer(SetupMixin):
    @abstractmethod
    def featurize(
        self,
        message_data: Dict[Text, Any],
        attributes: Optional[List[Text]] = None,
        excluded_attributes: Optional[List[Text]] = None,
    ) -> Dict[Text, List[Features]]:
        self.raise_if_setup_is(False)
        attributes = from_given_choose_supported_and_remove_excluded(
            given=message_data.keys(),
            supported=attributes,
            excluded=excluded_attributes,
        )
        return self._featurize(message_data, attributes)


class MessageDataFeaturizerUsingMultiHotVectors(MessageDataFeaturizer):
    def __init__(self, domain: Optional[Domain] = None,) -> None:
        self._item2index_mappings = None
        if domain is not None:
            self.setup(domain)

    def setup(self, domain: Domain) -> None:
        self.raise_if_setup_is(True)
        self._item2index_mappings = {
            key: item2index_mapping(items)
            for key, items in [
                (INTENT, domain.intents),
                (ACTION_NAME, domain.action_names_or_texts),
                (ENTITIES, domain.entity_states),
                (SLOTS, domain.slot_states),
                (ACTIVE_LOOP, domain.form_names),
            ]
        }

    def is_setup(self) -> bool:
        return self.item2index_mappings is not None

    def _encoding(self, message_data: Message, attribute: Text):
        """Turns the given mapping from a categorical variable name to a value.

        Note that the space of possible cateorical variables is determined by the
        domain that was used to setup this featurizer.

        # NOTE: this was `_state_features_for_attribute`
        """
        if attribute in {INTENT, ACTION_NAME}:
            return {message_data[attribute]: 1}  # type: ignore[dict-item]
        elif attribute == ENTITIES:
            # TODO/FIXME: no bilou tagging handled here, in contrast
            # featurize_entities
            return {entity: 1 for entity in message_data.get(ENTITIES, [])}
        elif attribute == ACTIVE_LOOP:
            return {message_data["name"]: 1}  # type: ignore[dict-item]
        elif attribute == SLOTS:
            return {
                f"{slot_name}_{i}": value
                for slot_name, slot_as_feature in message_data.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self.item2index_mappings.keys()}'."
            )

    def featurize_attribute(self, message_data: Message, attribute: Text) -> Features:
        """Creates a multi-hot encoding like feature for the given attribute.

        """
        if attribute not in message_data:
            raise ValueError(f"Expected {attribute} to be attribute of given substate.")

        encoding: Dict[Text, int] = self._encoding(message_data, attribute)

        # convert to a sparse matrix
        dim = len(self._item2index_mappings[attribute])
        row = np.zeros(dim, dtype=int)
        col = np.zeros(dim, dtype=int)
        data = np.zeros(dim, dtype=float)
        for feature_name, value in encoding.items():
            if feature_name in self.item2index_mapping:
                col.append(self.item2index_mappings[feature_name])
                data.append(value)
        features = scipy.sparse.coo_matrix((data, (row, col)))

        return Features(
            features,
            FEATURE_TYPE_SENTENCE,
            self.attribute,
            origin=self.__class__.__name__,
        )

    def _featurize(
        self, message_data: Dict[Text, Any], attributes: Optional[List[Text]] = None,
    ) -> Dict[Text, List[Features]]:
        self.raise_if_setup_is(False)

        attributes = from_given_choose_supported_and_remove_excluded(
            given=message_data.keys(),
            supported=self.item2index_mappings.keys(),
            excluded=None,
        )
        return {
            attribute: self.featurize_attribute(message_data, attribute)
            for attribute in attributes
        }


class MessageDataFeaturizerUsingInterpreter(MessageDataFeaturizer):
    def __init__(
        self,
        use_regex_interpreter: Optional[bool] = None,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
        bilou_tagging: bool = False,
    ) -> None:
        self.interpreter_handler = InterpreterHandler(
            use_regex_interpreter=use_regex_interpreter, interpreter=interpreter,
        )
        self.bilou_tagging = bilou_tagging
        self.encoding_spec = None

    def setup(self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]):
        self.interpreter_handler.set_or_fallback(interpreter)
        self._setup_entity_tagging(domain)

    def _setup_entity_tagging(self, domain: Domain):
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

    def is_setup(self) -> bool:
        return self.interpreter_handler.is_setup() and (self.encoding_spec is not None)

    def _featurize(
        self, message_data: Dict[Text, Any], attributes: List[Text]
    ) -> Dict[Text, List[Features]]:

        message = Message(data=message_data)
        parsed_message = self.interpreter.featurize_message(message)

        if self.bilou_tagging:
            bilou_utils.apply_bilou_schema_to_message(message)

        attribute_to_features: Dict[Text, List[Features]] = dict()
        for attr in attributes:
            all_features = parsed_message.get_sparse_features(
                attr
            ) + parsed_message.get_dense_features(attr)
            for features in all_features:
                if features is not None:
                    attribute_to_features[attr].append(features)

        if ENTITIES in attributes:
            attribute_to_features[ENTITY_TAGS] = [
                model_data_utils.get_tag_ids(
                    message, self.encoding_specs, self.bilou_tagging
                )
            ]

        return attribute_to_features


class InterpreterHandler(SetupMixin):
    def __init__(
        self,
        use_regex_interpreter: Optional[bool] = None,
        interpreter: Optional[NaturalLanguageInterpreter] = None,
    ) -> None:
        self._use_regex_interpreter = use_regex_interpreter
        self._interpreter = interpreter

    def is_setup(self):
        return self._interpreter is not None

    def set_or_fallback(
        self, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        if self._use_regex_interpreter is None:
            if interpreter is None:
                self._use_regex_interpreter = True
            else:
                self._use_regex_interpreter = isinstance(interpreter, RegexInterpreter)
        elif interpreter is None or (
            self._use_regex_interpreter
            and not isinstance(interpreter, RegexInterpreter)
        ):
            interpreter = RegexInterpreter()
            logger.debug("Fallback to RegexInterpreter.")
        self._interpreter = interpreter

    def get(self) -> NaturalLanguageInterpreter:
        self.raise_if_setup_is(False)
        return self._interpreter
