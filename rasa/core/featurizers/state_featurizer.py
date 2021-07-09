import logging
from rasa.shared.nlu.training_data.features import Features
import numpy as np
import scipy.sparse
from typing import List, Optional, Dict, Text, Any, Iterable

# from rasa.nlu.extractors.extractor import EntityTagSpec
# from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
# from rasa.nlu.utils import bilou_utils
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

logger = logging.getLogger(__name__)


def item2index_mapping(unique_items: Iterable[Text]) -> Dict[Text, int]:
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


def convert_sparse_sequence_to_sentence_features(
    features: List["Features"],
) -> List["Features"]:
    """Grabs all sparse sequence features and turns them into sparse sentence features.

    That is, this function filters all sparse sequence features and then
    turns each of these sparse sequence features into sparse sentence feature
    by summing over their first axis, respectively.

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


class StateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(
        self,
        item2index_mappings: Optional[Dict[Text, Any]] = None,
        action_texts: Optional[List[Text]] = None,
        use_regex_interpreter: Optional[bool] = None,
    ) -> None:
        """Initialize the single state featurizer."""
        inputs = [item2index_mappings, action_texts, use_regex_interpreter]
        if all(input is None for input in inputs) or all(
            input is None for input in inputs
        ):
            raise ValueError("Expected all or none inputs to be != None.")
        self.item2index_mappings = item2index_mappings
        self.action_texts = action_texts
        self.use_regex_interpreter = use_regex_interpreter

    def setup(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        """
        NOTE: domain and interpreter might not be known during instantiation

        Raises:
          RuntimeError if it has already been setup (cf. `is_ready`)
        """
        self.raise_if_ready_is(True)
        self.use_regex_interpreter = (
            isinstance(interpreter, RegexInterpreter) if interpreter else False
        )
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
        # # the following are used in featurize_target* functions *only*
        # self.action_texts = domain.action_texts
        # self.setup_featurization_of_target_entities(domain)

    # def setup_featurization_of_target_entities(self, domain: Domain):
    #     entities = sorted(domain.entity_states)
    #     if self.bilou_tagging:
    #         tag_id_index_mapping = {
    #             f"{prefix}{tag}": idx_1 * len(BILOU_PREFIXES) + idx_2 + 1
    #             for tag, idx_1 in enumerate(entities)
    #             for idx_2, prefix in enumerate(BILOU_PREFIXES)
    #         }
    #     else:
    #         tag_id_index_mapping = {
    #             tag: idx + 1  # +1 to keep 0 for the NO_ENTITY_TAG
    #             for tag, idx in enumerate(entities)
    #         }

    #     # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
    #     # needed for correct prediction for padding
    #     tag_id_index_mapping[NO_ENTITY_TAG] = 0

    #     # TODO
    #     #  The entity states used to create the tag-idx-mapping contains the
    #     #  entities and the concatenated entity and roles/groups. We do not
    #     #  distinguish between entities and roles/groups right now.
    #     #  we return a list to anticipate that
    #     self.encoding_spec = EntityTagSpec(
    #         tag_name=ENTITY_ATTRIBUTE_TYPE,
    #         tags_to_ids=tag_id_index_mapping,
    #         ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
    #         num_tags=len(tag_id_index_mapping),
    #     )

    def is_ready(self) -> bool:
        """
        returns true
        """
        return self.item2index_mappings is not None

    def raise_if_ready_is(self, should_be_ready: bool) -> None:
        if not self.is_ready() and should_be_ready:
            raise RuntimeError(
                f"Expected {self.__class__.__name__} has been `setup()`."
            )
        elif self.is_ready() and not should_be_ready:
            raise RuntimeError(
                f"Expected {self.__class__.__name__} had not been `setup()` before."
            )

    def check_and_replace_interpreter_if_needed(
        self, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> NaturalLanguageInterpreter:
        # TODO: Mismatch between train and test time is always a problem.. also with
        # different NLUInterpreters. We need a solution that captures that as well...
        if self.use_regex_interpreter and not isinstance(interpreter, RegexInterpreter):
            io_utils.warn(
                "Expected a regex featurizer. Falling back to new RegexInterpreter."
            )
            interpreter = RegexInterpreter()
        return interpreter

    def featurize_state(
        self, state: State, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        """Encode the given state with the help of the given interpreter.

        Args:
            state: The state to encode
            interpreter: The interpreter used to encode the state

        Returns:
            A dictionary of state_type to list of features.
        """
        self.raise_if_ready_is(False)
        interpreter = self.check_and_replace_interpreter_if_needed(interpreter)

        state_features = {}
        for substate_type, sub_state in state.items():

            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            name_attribute = state_utils.get_name_attribute(sub_state)

            if substate_type == PREVIOUS_ACTION:
                substate_features = self.featurize_substate_via_interpreter(
                    sub_state,
                    interpreter,
                    exclude={ENTITIES},
                    keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    use_fallback_interpreter={name_attribute},
                    fallback_is_sparse=True,
                )

            elif substate_type == USER and state_utils.previous_action_was_listen(
                state
            ):
                # featurize user only if it is "real" user input,
                # i.e. input from a turn after action_listen
                substate_features = self.featurize_substate_via_interpreter(
                    sub_state,
                    interpreter,
                    exclude={ENTITIES},
                    keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    use_fallback_interpreter={name_attribute},
                    fallback_is_sparse=True,
                )
                if sub_state.get(ENTITIES):
                    state_features[
                        ENTITIES
                    ] = self.featurize_substate_attribute_via_fallback_interpreter(
                        sub_state=sub_state, attribute=ENTITIES,
                    )

            elif substate_type in {SLOTS, ACTIVE_LOOP}:
                substate_features = {
                    substate_type: self.featurize_substate_attribute_via_fallback_interpreter(
                        sub_state=sub_state, attribute=substate_type,
                    )
                }
            else:
                substate_features = {}

            state_features.update(substate_features)

        return state_features

    def featurize_substate_via_interpreter(
        self,
        sub_state: SubState,
        interpreter: NaturalLanguageInterpreter,
        exclude: Optional[List[Text]] = None,
        keep_only_sparse_seq_converted_to_sent: Optional[List[Text]] = None,
        use_fallback_interpreter: Optional[List[Text]] = None,
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:
        """Featurize all attributes associated with a sub_state.


        # NOTE: this was `_extract_state_features`

        Args:
          sub_state: the sub-state to featurize
          interpreter: an optional interpreter to use instead of the default one
          keep_only_sparse_seq_converted_to_sent: Attributes for which the features
            obtained via the interpreter are post-processed. All sparse sequence
            features are turned into sparse sentence features (via summation) and only
            these resulting features will be returned.
            However, if there are no such features, then the attributes will not be
            included in the resulting feature dictionary).
          use_fallback_interpreter: Attributes for which `Features` will be created
            on the fly iff the interpreter did not create any features for them.
          sparse: Determines whether the features generated by
            `use_fallback_interpreter` are sparse.
        """
        self.raise_if_ready_is(False)
        interpreter = self.check_and_replace_interpreter_if_needed(interpreter)

        # (1) featurize the message/sub_state...
        message = Message(data=sub_state)
        parsed_message = interpreter.featurize_message(message)
        # TODO: is it clear that this works? if not, it should be made explicit
        # somewhere somehow .... -> SubState and Message dataclass things....

        # (2) gather all features from the featurized message -- except the ones
        # we are told to `exclude`
        exclude = exclude or []
        attributes = set(
            attribute for attribute in sub_state.keys() if attribute not in exclude
        )
        attribute_to_features: Dict[Text, List[Features]] = dict()
        for attribute in attributes:
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
        attribute_list = keep_only_sparse_seq_converted_to_sent or []
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
        attribute_list = use_fallback_interpreter or []
        for attribute in use_fallback_interpreter:
            if attribute not in attribute_to_features:
                attribute_to_features[
                    attribute
                ] = self.featurize_substate_attribute_via_fallback_interpreter(
                    sub_state, attribute, sparse=sparse,
                )
        return attribute_to_features

    def featurize_substate_attribute_via_fallback_interpreter(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> List["Features"]:
        """Creates a multi-hot encoding like feature for the given attribute.

        Leverages the item2index mappings computed during `prepare_training`.

        FIXME/TODO: point out clearly in docs that values for slots can be > 1
           (happens in other places to - cf. count vectorizer *except* sequence features
           generated from word-level vectorizers)

        NOTE: this was ` _create_features`
        """
        self.raise_if_ready_is(False)

        encoding: Dict[Text, int] = self._fallback_interpreter(sub_state, attribute)

        features = np.zeros(len(self.item2index_mappings[attribute]), np.float32)
        for feature_name, value in encoding.items():
            # check that the value is in default_feature_states to be able to assign
            # its value
            if feature_name in self.item2index_mappings[attribute]:
                features[self.item2index_mappings[attribute][feature_name]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        return [
            Features(
                features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
            )
        ]

    def _fallback_interpreter(
        self, sub_state: SubState, attribute: Text
    ) -> Dict[Text, int]:
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

    # def featurize_target_action_via_interpreter(
    #     self, action: Text, interpreter: Optional[NaturalLanguageInterpreter] = None,
    # ) -> Dict[Text, "Features"]:
    #     """

    #     TODO: must be possible to get rid of the interpreter check here / pass the
    #     "correct" interpreter right away...

    #     """
    #     self.raise_if_not_ready()
    #     interpreter = self.state_featurizer.check_and_replace_interpreter_if_needed(
    #         interpreter
    #     )

    #     sub_state = state_utils.create_substate_from_action(
    #         action, as_text=(action in self.action_texts)
    #     )
    #     return self.state_featurizer.featurize_substate_via_interpreter(
    #         sub_state, interpreter=interpreter
    #     )

    # def featurize_target_entity_via_interpreter(
    #     self,
    #     entity_data: Dict[Text, Any],
    #     interpreter: Optional[NaturalLanguageInterpreter] = None,
    # ) -> Dict[Text, "Features"]:
    #     """Encode the given entity data with the help of the given interpreter.

    #     Produce numeric entity tags for tokens.

    #     Args:
    #         entity_data: The dict containing the text and entity labels and locations
    #         interpreter: The interpreter used to encode the state
    #         #bilou_tagging: indicates whether BILOU tagging should be used or not

    #     Returns:
    #         A dictionary of entity type to list of features.

    #     TODO: must be possible to get rid of the interpreter check here / pass the
    #     "correct" interpreter right away...
    #     """
    #     self.raise_if_not_ready()
    #     interpreter = self.state_featurizer.check_and_replace_interpreter_if_needed(
    #         interpreter
    #     )

    #     # TODO
    #     #  The entity states used to create the tag-idx-mapping contains the
    #     #  entities and the concatenated entity and roles/groups. We do not
    #     #  distinguish between entities and roles/groups right now.
    #     if (
    #         not entity_data
    #         or not self.encoding_specs
    #         or self.encoding_specs.num_tags < 2
    #     ):
    #         # we cannot build a classifier with fewer than 2 classes
    #         return {}

    #     message = interpreter.featurize_message(Message(entity_data))

    #     if not message:
    #         return {}

    #     if self.bilou_tagging:
    #         bilou_utils.apply_bilou_schema_to_message(message)

    #     return {
    #         ENTITY_TAGS: [
    #             model_data_utils.get_tag_ids(
    #                 message, self.encoding_specs, self.bilou_tagging
    #             )
    #         ]
    #     }
