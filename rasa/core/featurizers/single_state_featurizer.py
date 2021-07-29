import logging
import copy
from re import sub
from rasa.utils.tensorflow.constants import SENTENCE
import numpy as np
import scipy.sparse
from typing import Any, Iterable, List, Optional, Dict, Text, Set, Union, TypedDict
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
    FEATURE_TYPE_SEQUENCE,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow import model_data_utils

logger = logging.getLogger(__name__)


def create_item2index_mapping(unique_items: Iterable[Text]) -> Dict[Text, int]:
    """Computes a mapping that maps each given label to a unique int index.
    The computed mapping is consistent across sessions since the unique_items are
    sorted before the mappin is computed.

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


def create_sparse_sentence_from_sparse_sequence_features(
    all_features: List[Features],
) -> List[Features]:
    """Creates sparse sentence features via summation of sparse sequence features.

    Args:
        all_features: list of Features that possibly does not contain any sparse
        sequence features
    Returns:
        a list with as many sparse sentence features as there are sparse sequence
        features in the given `all_features` list
    """
    return [
        Features(
            scipy.sparse.coo_matrix(feature.features.sum(0)),
            FEATURE_TYPE_SENTENCE,
            feature.attribute,
            feature.origin,
        )
        for feature in all_features
        if feature.is_sparse and feature.type == FEATURE_TYPE_SEQUENCE
    ]


def copy_featurized_message(
    featurized_message: Message, attributes: List[Text]
) -> Message:
    """Creates a copy that only contains the specified attributes.
    """
    featurized_message = copy.deepcopy(featurized_message)
    attributes_to_be_removed = set(featurized_message.data.keys()).difference(
        attributes
    )
    for attribute in attributes_to_be_removed:
        del featurized_message.data[attribute]
    featurized_message.features = [
        feat
        for feat in featurized_message.features
        if feat.attribute not in attributes_to_be_removed
    ]
    return featurized_message


def lookup_featurized_message_for_substate(
    sub_state: Dict[Text, Any], e2e_features: Optional[Dict[Text, Message]] = None,
) -> Message:
    """Attempts to ....

    TODO: there was a function for that in the prototype somewhere, replace this
    """
    key = next(
        k for k in sub_state.keys() if k in {ACTION_NAME, ACTION_TEXT, INTENT, TEXT}
    )
    # TODO: instead of dict, we need component that holds a cache and can
    # fallback to on-the-fly computation of features?
    if e2e_features:
        featurized_message = e2e_features[key]
    else:
        featurized_message = Message(data=sub_state)
        # = unfeaturized message, TODO: needed for Memoization (?)
    return featurized_message


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
        self.is_setup = False

    def setup(self, domain: Domain, bilou_tagging: bool = False,) -> None:
        """Gets necessary information for featurization from domain.

        Args:
            domain: An instance of :class:`rasa.shared.core.domain.Domain`.
            bilou_tagging: indicates whether BILOU tagging should be used or not
        """
        self._item2index_mappings = {
            key: create_item2index_mapping(items)
            for key, items in [
                (INTENT, domain.intents),
                (ACTION_NAME, domain.action_names_or_texts),
                (ENTITIES, domain.entity_states),
                (SLOTS, domain.slot_states),
                (ACTIVE_LOOP, domain.form_names),
            ]
        }
        self.action_texts = domain.action_texts
        self.entity_tag_specs = self._setup_entity_tag_specs(bilou_tagging)
        self.is_setup = True

    def _setup_entity_tag_specs(
        self, bilou_tagging: bool = False
    ) -> List[EntityTagSpec]:
        """Returns the tag to index mapping for entities.

        Returns:
            Tag to index mapping.
        """
        if ENTITIES not in self._item2index_mappings:
            return []

        if bilou_tagging:
            tag_id_index_mapping = {
                f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
                for tag, idx_1 in self._item2index_mappings[ENTITIES].items()
                for idx_2, prefix in enumerate(BILOU_PREFIXES)
            }
        else:
            tag_id_index_mapping = {
                tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
                for tag, idx in self._item2index_mappings[ENTITIES].items()
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

    StateEncoding = TypedDict(
        "StateEncoding",
        {
            "input": Dict[Text, List["Features"]],
            "target_entity": Dict[Text, List["Features"]],
            "target_action": int,
        },
        total=False,
    )

    def encode_state(
        self,
        state: State,
        domain: Domain,
        bilou_tagging: bool = False,
        e2e_features: Optional[Dict[Text, Message]] = None,
        targets_include_entities: bool = False,
        targets_include_actions: bool = False,
        remove_user_text_if_intent: bool = True,
    ) -> StateEncoding:
        """Encode the given state.

        Args:
            state: The state to encode
            bilou Tagging
            e2e_features: A mapping of message text to parsed message with e2e features

        Returns:
            A dictionary of state_type to list of features.
        """
        if not self.is_setup:
            raise RuntimeError(f"{self.__class__.__name__} needs to be setup() first.")

        features_input = {}
        features_target_entity = {}
        features_target_action = -1

        for state_type, sub_state in state.items():

            # (1) previous_action
            if state_type == PREVIOUS_ACTION:

                featurized_message = lookup_featurized_message_for_substate(
                    sub_state, e2e_features
                )
                attributes2features = self._extract_features_from_featurized_message(
                    featurized_message, None,  # i.e. all # FIXME: text / name ?
                )

                if ACTION_NAME in sub_state:
                    self._special_handling_of_attribute(
                        sub_state=sub_state,
                        extracted_features=attributes2features,
                        special_attribute=ACTION_NAME,
                    )
                features_input.update(attributes2features)

                # action targets are extracted from action sub_state
                if targets_include_actions:
                    features_target_action = domain.index_for_action(
                        sub_state.get(ACTION_NAME, None) or sub_state[ACTION_TEXT]
                    )

            # (2) user
            elif state_type == USER and is_prev_action_listen_in_state(state):
                # featurize user sub-state only if it is "real" user input,
                # i.e. input from a turn after action_listen

                featurized_message = lookup_featurized_message_for_substate(
                    sub_state, e2e_features
                )

                if remove_user_text_if_intent and INTENT in sub_state:
                    # TODO: just add an include/exclude parameters to the "extract"
                    # functions instead of this workaround...
                    attributes = set(featurized_message.data.keys()).difference(TEXT)
                    featurized_message = copy_featurized_message(
                        featurized_message, attributes
                    )

                attributes2features = self._extract_features_from_featurized_message(
                    featurized_message=featurized_message,
                    attributes=set(
                        attribute
                        for attribute in featurized_message.data.keys()
                        if attribute != ENTITIES
                    ),
                )

                if INTENT in sub_state:
                    self._special_handling_of_attribute(
                        sub_state=sub_state,
                        extracted_features=attributes2features,
                        special_attribute=INTENT,
                    )

                if ENTITIES in sub_state:
                    multihot_feats_for_entities = self._create_multihot_vector_features(
                        sub_state, ENTITIES, sparse=True
                    )
                    attributes2features[ENTITIES] = multihot_feats_for_entities
                features_input.update(attributes2features)

                # entity targets are extracted from user sub_state
                if targets_include_entities:
                    # NOTE: this requires tokens from the featurized message
                    features_target_entity = self._extract_entity_features_from_featurized_message(
                        featurized_message, bilou_tagging
                    )

            # (3+4) slot and active_loop
            elif state_type in {SLOTS, ACTIVE_LOOP}:
                features_input[state_type] = self._create_multihot_vector_features(
                    sub_state, state_type, sparse=True
                )

        # FIXME: constants
        output = {"input": features_input}
        if targets_include_actions:
            output["target_action"] = features_target_action
        if targets_include_entities:
            output["target_entity"] = features_target_entity
        return output

    def encode_all_actions(
        self, domain: Domain, e2e_features: Dict[Text, Message],
    ) -> List[Dict[Text, List["Features"]]]:
        """Encode all action from the domain. Only e2e features will be used.

        Args:
            domain: The domain that contains the actions.
            e2e_features: A mapping of message text to parsed message with e2e features

        Returns:
            a list of

        Raises:
            RuntimeError if any of the action texts or names from the `domain` cannot
            be featurized via the featurized messages in the `e2e_features`.
        """
        features = []
        for attribute, actions in [
            (ACTION_NAME, domain.action_names_without_texts),
            (ACTION_TEXT, domain.action_texts),
        ]:
            for action in actions:
                # NOTE: this is tricky and only works iff the sub_state looks exactly
                # like this during lookup table computation
                dummy_state = {PREVIOUS_ACTION: {attribute: action}}
                state_encoding = self.encode_state(
                    state=dummy_state,
                    domain=domain,
                    bilou_tagging=False,
                    e2e_features=e2e_features,
                    targets_include_actions=False,
                    targets_include_entities=False,
                    remove_user_text_if_intent=False,
                )
                features_for_action = state_encoding["input"]
                if not features_for_action:
                    raise RuntimeError(
                        f"Could not lookup features for state {dummy_state}."
                    )
                features.append(features_for_action)
        return features

    def _extract_features_from_featurized_message(
        self, featurized_message: Optional[Message], attributes: Optional[Set[Text]],
    ) -> Dict[Text, List[Features]]:
        """Extracts all of the features for the given attributes from the message.

        Args:
           featurized_message: a message that possibly contains some `Features`
           attributes: all attributes to be considered during extraction
        Returns:
           dictionary mapping all attributes for which `Features` were present in the
           given `featurized_message` to a list containing those `Features`
        """
        if featurized_message is None:
            return {}
        attributes = attributes or set(featurized_message.data.keys())
        output = dict()
        for attribute in attributes:
            all_features = featurized_message.get_sparse_features(
                attribute
            ) + featurized_message.get_dense_features(attribute)
            for features in all_features:
                if features is not None:
                    output.setdefault(attribute, []).append(features)
        return output

    def _extract_entity_features_from_featurized_message(
        self, featurized_message: Message, bilou_tagging: bool,
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
            featurized_message.data.get(ENTITIES)
            and not self.entity_tag_specs
            and self.entity_tag_specs[0].num_tags >= 2
        ):
            # we cannot build a classifier with fewer than 2 classes
            if bilou_tagging:
                bilou_utils.apply_bilou_schema_to_message(featurized_message)

            entity_features = {
                ENTITY_TAGS: [
                    model_data_utils.get_tag_ids(
                        featurized_message, self.entity_tag_specs[0], bilou_tagging
                    )
                ]
            }
        return entity_features

    def _special_handling_of_attribute(
        self,
        sub_state: Dict[Text, Any],
        extracted_features: Dict[Text, List[Features]],
        special_attribute: Text,
    ) -> None:
        """Modifies the `extracted_features` for `special_attribute` in-place.

        (1) if features for the `special_attribute` are contained in the given
           `extracted_features`, then this method will
           - first try *replace* them with sparse sentence features
               created via summation from sparse sequence features and
           - if that fail, attempt to use the sparse sentence features already
             present in the `extracted_features` and
           - if there are no sparse features at all, continue with step (2)

        (2) if there were not features present (or if there where features but no sparse
            features) for the `special_attribute`, some multi-hot vector features
            will be created from scratch.

        Example:
            E.g. NLU pipeline might not create features for user or action.
            This might happen, for example, when we have action_name in the state
            but it did not get featurized because only a character level
            CountVectorsFeaturizer was included in the config.
            # TODO: example is from old comment. is this still correct?
        """

        if special_attribute in extracted_features:
            sparse_sentence_features = create_sparse_sentence_from_sparse_sequence_features(
                extracted_features[special_attribute],
            )
            if len(sparse_sentence_features) == 0:
                # FIXME: this is different from before but it makes sense .... ?
                sparse_sentence_features = [
                    feat
                    for feat in extracted_features[special_attribute]
                    if feat.is_sparse and feat.type == SENTENCE
                ]

            extracted_features[special_attribute] = sparse_sentence_features

        if special_attribute and special_attribute not in extracted_features:
            extracted_features[special_attribute] = [
                self._create_multihot_vector_feature(
                    sub_state=sub_state, attribute=special_attribute
                )
            ]

    def _prepare_multihot_vector_features_for_attribute(
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
                f"It must be one of '{self._item2index_mappings.keys()}'."
            )

    def _create_multihot_vector_feature(
        self, sub_state: SubState, attribute: Text,
    ) -> Features:
        """Creates a multi-hot encoding like feature for the given attribute.

        Args:
          sub_state:
          attribute:
        Returns:
          a sparse sentence feature
        """
        # TODO: this could become a Featurizer graph component but might not be
        # worth to include this in lookup (?)
        if attribute not in sub_state:
            raise ValueError(
                f"Expected {attribute} to be attribute of given substate {sub_state}."
            )

        encoding: Dict[
            Text, int
        ] = self._prepare_multihot_vector_features_for_attribute(sub_state, attribute)

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
