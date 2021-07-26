import logging
from operator import sub
from os import stat
import numpy as np
from numpy.lib.function_base import copy
import scipy.sparse
from typing import List, Optional, Dict, Text, Set, Union
from collections import defaultdict

from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.utils import bilou_utils
from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.core.constants import PREVIOUS_ACTION, ACTIVE_LOOP, USER, SLOTS
from rasa.shared.core.trackers import is_prev_action_listen_in_state
from rasa.shared.nlu.constants import (
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
    TEXT,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow import model_data_utils

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(self) -> None:
        """Initialize the single state featurizer."""
        self._default_feature_states = {}
        self.action_texts = []
        self.entity_tag_specs = []

    def _create_entity_tag_specs(
        self, bilou_tagging: bool = False
    ) -> List[EntityTagSpec]:
        """Returns the tag to index mapping for entities.

        Returns:
            Tag to index mapping.
        """
        if ENTITIES not in self._default_feature_states:
            return []

        if bilou_tagging:
            tag_id_index_mapping = {
                f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
                for tag, idx_1 in self._default_feature_states[ENTITIES].items()
                for idx_2, prefix in enumerate(BILOU_PREFIXES)
            }
        else:
            tag_id_index_mapping = {
                tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
                for tag, idx in self._default_feature_states[ENTITIES].items()
            }

        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_index_mapping[NO_ENTITY_TAG] = 0

        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        #  we return a list to anticipate that
        return [
            EntityTagSpec(
                tag_name=ENTITY_ATTRIBUTE_TYPE,
                tags_to_ids=tag_id_index_mapping,
                ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
                num_tags=len(tag_id_index_mapping),
            )
        ]

    def prepare_for_training(
        self, domain: Domain, bilou_tagging: bool = False,
    ) -> None:
        """Gets necessary information for featurization from domain.

        Args:
            domain: An instance of :class:`rasa.shared.core.domain.Domain`.
            bilou_tagging: indicates whether BILOU tagging should be used or not
        """

        # store feature states for each attribute in order to create binary features
        def convert_to_dict(feature_states: List[Text]) -> Dict[Text, int]:
            return {
                feature_state: idx for idx, feature_state in enumerate(feature_states)
            }

        self._default_feature_states[INTENT] = convert_to_dict(domain.intents)
        self._default_feature_states[ACTION_NAME] = convert_to_dict(
            domain.action_names_or_texts
        )
        self._default_feature_states[ENTITIES] = convert_to_dict(domain.entity_states)
        self._default_feature_states[SLOTS] = convert_to_dict(domain.slot_states)
        self._default_feature_states[ACTIVE_LOOP] = convert_to_dict(domain.form_names)
        self.action_texts = domain.action_texts
        self.entity_tag_specs = self._create_entity_tag_specs(bilou_tagging)

    def _prepare_multihot_encoding_for_attribute(
        self, sub_state: SubState, attribute: Text
    ) -> Dict[Text, int]:
        # FIXME: the code below is not type-safe, but fixing it
        #        would require more refactoring, for instance using
        #        data classes in our states
        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}  # type: ignore[dict-item]
        elif attribute == ENTITIES:
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
                f"It must be one of '{self._default_feature_states.keys()}'."
            )

    def _create_multihot_vector_features(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> List["Features"]:
        # TODO: this could become a Featurizer graph component but might not be
        # worth to include this in lookup (?)
        input_features = self._prepare_multihot_encoding_for_attribute(
            sub_state, attribute
        )

        # FIXME: this should be sparse from the start / only used with sparse=True
        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in input_features.items():
            # check that the value is in default_feature_states to be able to assign
            # its value
            if state_feature in self._default_feature_states[attribute]:
                features[self._default_feature_states[attribute][state_feature]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        return [
            Features(
                features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
            )
        ]

    @staticmethod
    def _to_sparse_sentence_features(
        sparse_sequence_features: List["Features"],
    ) -> List["Features"]:
        return [
            Features(
                scipy.sparse.coo_matrix(feature.features.sum(0)),
                FEATURE_TYPE_SENTENCE,
                feature.attribute,
                feature.origin,
            )
            for feature in sparse_sequence_features
        ]

    @staticmethod
    def _get_partial_message(
        parsed_message: Message, attributes: List[Text]
    ) -> Message:
        parsed_message = copy.deepcopy(parsed_message)
        attributes_to_be_removed = set(parsed_message.data.keys()).difference(
            attributes
        )
        for attribute in attributes_to_be_removed:
            del parsed_message.data[attribute]
        parsed_message.features = [
            feat
            for feat in parsed_message.features
            if feat.attribute not in attributes_to_be_removed
        ]
        return parsed_message

    @staticmethod
    def _get_partial_state(
        state: State,
        sub_state_type: Text,
        sub_state_type_attribute_combinations: Optional[Dict[Text, List[Text]]] = None,
    ) -> State:
        """
        Returns:
          a deep copy if there is some sub_state_type_attribute_combinations given,
          otherwise just a reference to the given state is returned
        """
        if sub_state_type_attribute_combinations is None:
            return state
        partial_state = dict()
        for sub_state_type, attributes in sub_state_type_attribute_combinations.items():
            sub_state = dict()
            if sub_state_type in state and attributes:
                attributes_left = set(sub_state.keys()).intersection(attributes)
                for attribute in attributes_left:
                    sub_state[attribute] = copy.deepcpoy(
                        state[sub_state_type][attribute]
                    )
                partial_state[sub_state] = sub_state
        return partial_state

    def _get_features_from_parsed_message(
        self, parsed_message: Optional[Message], attributes: Set[Text]
    ) -> Dict[Text, List["Features"]]:
        if parsed_message is None:
            return {}

        output = defaultdict(list)
        for attribute in attributes:
            all_features = parsed_message.get_sparse_features(
                attribute
            ) + parsed_message.get_dense_features(attribute)

            for features in all_features:
                if features is not None:
                    output[attribute].append(features)

        # if features for INTENT or ACTION_NAME exist,
        # they are always sparse sequence features;
        # transform them to sentence sparse features
        if output.get(INTENT):
            output[INTENT] = self._to_sparse_sentence_features(output[INTENT])
        if output.get(ACTION_NAME):
            output[ACTION_NAME] = self._to_sparse_sentence_features(output[ACTION_NAME])

        return output

    def _extract_input_features(
        self, parsed_message: Message, name_attribute: Text, sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:
        """Extracts features of all attributes except entities - and computes fallbacks.

        Note that for the special name_attribute a multihot vector feature is
        created if no features could be extracted for this feature.
        TODO: does this really happen in the case described below and why?
        """

        # remove entities from possible attributes
        attributes = set(
            attribute
            for attribute in parsed_message.data.keys()
            if attribute != ENTITIES
        )
        output = self._get_features_from_parsed_message(parsed_message, attributes)

        # check that name attributes have features
        if name_attribute and name_attribute not in output:
            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            output[name_attribute] = self._create_multihot_vector_features(
                parsed_message.data, name_attribute, sparse
            )

        return output

    def _extract_entity_features(
        self, parsed_message: Message, bilou_tagging: bool
    ) -> Dict[Text, List[Features]]:
        """Extract entity features from the given message and apply bilou tagging.

        Note that features will only be extracted iff the entity_tag_spec says that
        there are at least 2 entities in total.

        Moreover, the entity states used to create the tag-idx-mapping contains the
        entities and the concatenated entity and roles/groups. We do not
        distinguish between entities and roles/groups right now.
        """
        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        entity_features = {}
        if (
            parsed_message.data.get(ENTITIES)
            and not self.entity_tag_specs
            and self.entity_tag_specs[0].num_tags >= 2
        ):
            # we cannot build a classifier with fewer than 2 classes
            if bilou_tagging:
                bilou_utils.apply_bilou_schema_to_message(parsed_message)

            entity_features = {
                ENTITY_TAGS: [
                    model_data_utils.get_tag_ids(
                        parsed_message, self.entity_tag_specs[0], bilou_tagging
                    )
                ]
            }
        return entity_features

    @staticmethod
    def _convert_action_names_or_texts_to_ids(
        action_names_or_texts: List[Text], domain: Domain
    ) -> np.ndarray:
        return np.array(
            [domain.index_for_action(action) for action in action_names_or_texts]
        )

    def _extract_action_features(
        self, parsed_message: Message,
    ) -> Dict[Text, List["Features"]]:
        # Note: no need to check anymore if the (action_name or action_text) is in
        # self.action_texts to decide how to call interpreter :)
        attribute = ACTION_TEXT if parsed_message.data.get(ACTION_TEXT) else ACTION_NAME
        features = self._get_features_from_parsed_message(parsed_message, [attribute])
        return features  # features[attribute]

    def encode_state(
        self,
        state: State,
        domain: Domain,
        bilou_tagging: bool = False,
        e2e_features: Optional[Dict[Text, Message]] = None,
        separate_entities: bool = False,
        separate_actions: bool = False,
        remove_user_text_if_intent: bool = True,
    ) -> Dict[Text, Union[Dict[Text, List["Features"]], np.ndarray]]:
        """Encode the given state.

        Args:
            state: The state to encode
            bilou Tagging
            e2e_features: A mapping of message text to parsed message with e2e features

        Returns:
            A dictionary of state_type to list of features.
        """
        input_features = {}
        entity_features = {}
        action_features = np.ndarray([], dtype=int)

        for state_type, sub_state in state.items():

            is_action_substate = state_type == PREVIOUS_ACTION
            # featurize user only if it is "real" user input,
            # i.e. input from a turn after action_listen
            is_user_substate = state_type == USER and is_prev_action_listen_in_state(
                state
            )

            if is_action_substate or is_user_substate:
                key = next(
                    k
                    for k in sub_state.keys()
                    if k in {ACTION_NAME, ACTION_TEXT, INTENT, TEXT}
                )
                # TODO: instead of dict, we need component that holds a cache and can
                # fallback to on-the-fly computation of features?
                if e2e_features:
                    parsed_message = e2e_features[key]
                else:
                    parsed_message = Message(data=sub_state)
                    # = unfeaturized message, TODO: needed for Memoization (?)
                assert parsed_message

                if is_action_substate:
                    input_features.update(
                        self._extract_input_features(
                            parsed_message, name_attribute=ACTION_NAME, sparse=True
                        )
                    )

                    if separate_actions:
                        action_features: np.ndarray = self._convert_action_names_or_texts_to_ids(
                            sub_state[ACTION_NAME] or sub_state[ACTION_TEXT], domain
                        )

                elif is_user_substate:

                    if remove_user_text_if_intent:
                        # NOTE: needs to happen before next step - the name_attribute
                        # == intent will trigger the creation of a multi-hot feature
                        # just like it would have in the previous version where
                        # we removed the TEXT if INTENT was present during the
                        # ... (now: `extract_states_from_trackers_for_training`)

                        # TODO: just add an include/exclude parameters to the "extract"
                        # functions instead of this workaround...
                        if INTENT in parsed_message.data:
                            attributes = set(parsed_message.data.keys()).difference(
                                TEXT
                            )
                            parsed_message = self._get_partial_message(
                                parsed_message, attributes
                            )

                    input_features.update(
                        self._extract_input_features(
                            parsed_message, name_attribute=INTENT, sparse=True
                        )
                    )
                    if parsed_message.get(ENTITIES):
                        input_features[
                            ENTITIES
                        ] = self._create_multihot_vector_features(
                            parsed_message.data, ENTITIES, sparse=True
                        )

                    if separate_entities:
                        entity_features = self._extract_entity_features(
                            parsed_message, bilou_tagging
                        )
                else:
                    raise ValueError("This should not happen.")

                # for user and action substates:

            if state_type in {SLOTS, ACTIVE_LOOP}:
                input_features[state_type] = self._create_multihot_vector_features(
                    sub_state, state_type, sparse=True
                )

        # FIXME: constants
        return {
            "input": input_features,
            "entity": entity_features,
            "action": action_features,
        }

    def encode_all_actions(
        self, domain: Domain, e2e_features: Optional[Dict[Text, Message]] = None,
    ) -> List[Dict[Text, List["Features"]]]:
        """Encode all action from the domain. Only e2e features will be used.

        Args:
            domain: The domain that contains the actions.
            e2e_features: A mapping of message text to parsed message with e2e features

        Returns:
            A list of encoded actions.
        """
        return [
            self._extract_action_features(e2e_features[action])
            for action in domain.action_names_or_texts
        ]
