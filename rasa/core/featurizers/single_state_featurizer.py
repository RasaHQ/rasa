from abc import abstractmethod
import logging
from os import name
import numpy as np
import scipy.sparse
from typing import List, Optional, Dict, Text, Set, Any, Iterable
from collections import defaultdict

import rasa.shared.utils.io
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.nlu.utils import bilou_utils
from rasa.nlu.utils.bilou_utils import BILOU_PREFIXES
from rasa.shared.core.domain import SubState, State, Domain
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.shared.core.constants import PREVIOUS_ACTION, ACTIVE_LOOP, USER, SLOTS
from rasa.shared.constants import DOCS_URL_MIGRATION_GUIDE
from rasa.shared.core.trackers import is_prev_action_listen_in_state
from rasa.shared.nlu.constants import (
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
    ACTION_TEXT,
    ACTION_NAME,
    FEATURE_TYPE_SEQUENCE,
    INTENT,
    NO_ENTITY_TAG,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
)
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
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


def get_name_attribute(sub_state: SubState) -> Optional[Text]:
    """ TODO: this could go into some SubState class....
    """
    # there is always either INTENT or ACTION_NAME
    name_attributes = [
        attribute
        for attribute in sub_state.keys()
        if attribute in {INTENT, ACTION_NAME}
    ]
    if len(name_attributes) == 0:
        raise ValueError("Expected INTENT or ACTION_NAME.")
    elif len(name_attributes) > 1:  # TODO: added this because the comment said so..
        raise ValueError("Expected either INTENT or ACTION_NAME, not both.")
    return name_attributes[0]


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


class SingleStateFeaturizer:
    """Base class to transform the dialogue state into an ML format.

    Subclasses of SingleStateFeaturizer will decide how a bot will
    transform the dialogue state into a dictionary mapping an attribute
    to its features. Possible attributes are: INTENT, TEXT, ACTION_NAME,
    ACTION_TEXT, ENTITIES, SLOTS and ACTIVE_LOOP. Each attribute will be
    featurized into a list of `rasa.utils.features.Features`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the single state featurizer."""
        self.prepared = False
        self.use_regex_interpreter = False

        # FIXME: If this is re-instantiated for a policy that we want to fine-tune
        # then the information to use a regex interpreter will be lost unless
        # the  NLU Interpreter is still not present during re-training....
        # Is this a problem?

    def prepare(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> None:
        """
        TODO: all of this looks like it could be done in the init....
        TODO/FIXME: the item2index_mappings are also needed during inference, which
          is why it is very confusing this is computed in "prepare_for_training" only...
        """
        # TODO: the following this was if...: = True before, is this a problem?
        self.use_regex_interpreter = isinstance(interpreter, RegexInterpreter)
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
        self.action_texts = domain.action_texts

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
        # Note: This method is called during both prediction and training.
        # `self._use_regex_interpreter == True` means that core was trained
        # separately. Therefore we have to replace the passed interpreter
        # (which is based on some trained nlu model)  with a default
        # RegexInterpreter (i.e. same as during training) to make sure
        # that prediction and train time features are the same

        # FIXME/TODO:
        # 1. Would be cool if users would be notified that they're not training
        #    the policy based on what their NLU pipeline returns.... (cf.
        #    featurize_via_interpreter)
        # 2. some featurization depends on the feature2index mappings with are *only*
        #    there if prepare_trianing has been called before - does this happen
        #    during inference? If yes, then that's a confusing name. If not, then
        #    why is it ok that features won't be created?

        state_features = {}
        for substate_type, sub_state in state.items():

            # nlu pipeline didn't create features for user or action
            # this might happen, for example, when we have action_name in the state
            # but it did not get featurized because only character level
            # CountVectorsFeaturizer was included in the config.
            name_attribute = get_name_attribute(sub_state)

            if substate_type == PREVIOUS_ACTION:
                substate_features = self._featurize_substate_via_interpreter(
                    sub_state,
                    interpreter,
                    exclude={ENTITIES},
                    keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    fallback_to_custom_encoding={name_attribute},
                    fallback_is_sparse=True,
                )

            elif substate_type == USER and is_prev_action_listen_in_state(state):
                # featurize user only if it is "real" user input,
                # i.e. input from a turn after action_listen
                substate_features = self._featurize_substate_via_interpreter(
                    sub_state,
                    interpreter,
                    exclude={ENTITIES},
                    keep_only_sparse_seq_converted_to_sent={INTENT, ACTION_NAME},
                    fallback_to_custom_encoding={name_attribute},
                    fallback_is_sparse=True,
                )
                if sub_state.get(ENTITIES):
                    state_features[
                        ENTITIES
                    ] = self._featurize_attribute_via_custom_encoding(
                        sub_state=sub_state, attribute=ENTITIES,
                    )

            elif substate_type in {SLOTS, ACTIVE_LOOP}:
                substate_features = {
                    substate_type: self._featurize_attribute_via_custom_encoding(
                        sub_state=sub_state, attribute=substate_type,
                    )
                }
            else:
                substate_features = {}

            state_features.update(substate_features)

        return state_features

    def featurize_action(
        self, action: Text, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:
        """

        """
        # FIXME: if prepare_for_training has not been called then action_texts will
        # be an empty list and this always defaults to "ACTION_NAME" - is this intended?
        key = ACTION_TEXT if action in self.action_texts else ACTION_NAME
        return self._featurize_substate_via_interpreter(
            sub_state={key: action}, interpreter=interpreter
        )

    # def featurize_all_actions # NOTE: we don't need that... use list gen....

    def _featurize_substate_via_interpreter(
        self,
        sub_state: SubState,
        interpreter: NaturalLanguageInterpreter,
        exclude: Optional[List[Text]] = None,
        keep_only_sparse_seq_converted_to_sent: Optional[List[Text]] = None,
        fallback_to_custom_encoding: Optional[List[Text]] = None,
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
          fallback_to_custom_encoding: Attributes for which `Features` will be created
            on the fly iff the interpreter did not create any features for them
            (or if there were no sparse features present for the attributes listed
            in `keep_only_sparse_seq_converted_to_sent` ).
          sparse: Determines whether the features generated by
            `fallback_to_custom_encoding` are sparse.
        """

        if self.use_regex_interpreter and not isinstance(interpreter, RegexInterpreter):
            interpreter = RegexInterpreter()

        # featurize the message/sub_state...
        # TODO: is it clear that this works? if not, it should be made explicit
        # somewhere somehow .... -> SubState and Message dataclass things....
        message = Message(data=sub_state)
        parsed_message = interpreter.featurize_message(message)  # TODO:

        # ... and gather all features except the ones for Entities
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

        attribute_list = keep_only_sparse_seq_converted_to_sent or []
        for attribute in attribute_list:
            if attribute_to_features.get(attribute):
                converted_features = convert_sparse_sequence_to_sentence_features(
                    attribute_to_features[attribute]
                )
                if converted_features:
                    attribute_to_features[attribute] = converted_features

        attribute_list = fallback_to_custom_encoding or []
        for attribute in fallback_to_custom_encoding:
            if attribute not in attribute_to_features:
                attribute_to_features[attribute] = self.featurize_substate_attribute(
                    sub_state, attribute, sparse=sparse,
                )
        return attribute_to_features

    def _featurize_attribute_via_custom_encoding(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> List["Features"]:
        """Creates a multi-hot encoding like feature for the given attribute.

        Leverages the item2index mappings computed during `prepare_training`.

        FIXME/TODO: would've expected a one-hot encoding type of thing, but for
          SLOTS the values are >1 ...(?!?)

        # NOTE: this was ` _create_features`

        """
        if not self.item2index_mappings:
            raise RuntimeError("Expected item2index_mapping to be non-empty.")

        encoding: Dict[Text, int] = self._custom_encoding(sub_state, attribute)

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

    def _custom_encoding(self, sub_state: SubState, attribute: Text) -> Dict[Text, int]:
        """Derives an encoding to be used by `_featurize_attribute_via_custom_encoding`.

        # NOTE: this was `_state_features_for_attribute`

        """
        # FIXME: Create dataclasses for states, substates, messages,...
        #   Then, e.g. the following would be type-safe.

        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}  # type: ignore[dict-item]
        elif attribute == ENTITIES:
            # TODO: no bilou tagging handled here, in contrast to EntityFeaturizer below
            # (which was part of SingleStateFeaturizer before)
            return {entity: 1 for entity in sub_state.get(ENTITIES, [])}
        elif attribute == ACTIVE_LOOP:
            return {sub_state["name"]: 1}  # type: ignore[dict-item]
        elif attribute == SLOTS:
            return {
                f"{slot_name}_{i}": value
                # FIXME/TODO: is this intended? sparse features will have these values as *entries*
                for slot_name, slot_as_feature in sub_state.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self.item2index_mappings.keys()}'."
            )


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    """Dialogue State featurizer which features the state as binaries."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Creates featurizer."""
        super().__init__()
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in Rasa Open Source 3.0.0. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

    def _featurize_substate_via_interpreter(
        self,
        sub_state: SubState,
        interpreter: NaturalLanguageInterpreter,
        exclude: List[Text],
        keep_only_sparse_seq_converted_to_sent: List[Text],
        fallback_to_custom_encoding: Optional[List[Text]],
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:
        """Ignores the intepreter and returns features derived from custom encodings.

        Compared to the regular SingleStateFeaturizer, the behaviour is the same as if
        the interpreter fails to produce any features and `fallback_to_custom_encoding`
        has been set to the named_attribute only.

        Args:
          substate:
          interpreter: ignored
          exclude: ignored
          keep_only_sparse_seq_converted_to_sent: ignored
          fallback_to_custom_encoding: ignored
          sparse: Determines whether the generated feature is sparse.
        """
        name_attribute = get_name_attribute(sub_state)
        if name_attribute:
            return {
                name_attribute: self._create_features(sub_state, name_attribute, sparse)
            }

        return {}


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # it is hard to fully mimic old behavior, but SingleStateFeaturizer
        # does the same thing if nlu pipeline is configured correctly
        rasa.shared.utils.io.raise_deprecation_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in Rasa Open Source 3.0.0. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            docs=DOCS_URL_MIGRATION_GUIDE,
        )


##### TODO/FIXME:
# - The following seems to be completely separate from the SingleStateFeaturizer
# - it implements a very similar yet different featurization


class EntityFeaturizer:
    def __init__(self, domain: Domain, bilou_tagging: bool = False,) -> None:
        """Initialize the single state featurizer."""

        self.domain = domain
        self.bilou_tagging = bilou_tagging
        self.encoding_spec = self._create_entity_tag_encoding_spec(
            bilou_tagging=self.bilou_tagging, domain=self.domain
        )

    @staticmethod
    def _create_entity_tag_encoding_spec(
        bilou_tagging: bool, domain: Domain,
    ) -> EntityTagSpec:
        """Returns an EntityTagSpec which includes e.g. an entity tag to index mapping.

        TODO: several todos in here :D
        """
        # sort to get a consistent mapping across sessions
        entities = sorted(domain.entity_states)

        if not entities:
            return []

        if bilou_tagging:
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
        return EntityTagSpec(
            tag_name=ENTITY_ATTRIBUTE_TYPE,
            tags_to_ids=tag_id_index_mapping,
            ids_to_tags={value: key for key, value in tag_id_index_mapping.items()},
            num_tags=len(tag_id_index_mapping),
        )

    def encode(
        self,
        entity_data: Dict[Text, Any],  # TODO/FIXME: this is message data...
        interpreter: NaturalLanguageInterpreter,
        # NOTE: removed bilou tagging parameter here because this
        # should fail if != self.bilou_tagging... (TODO: double-check)
    ) -> Dict[Text, List["Features"]]:
        """Encode the given entity data with the help of the given interpreter.

        Produce numeric entity tags for tokens.

        Args:
            entity_data: The dict containing the text and entity labels and locations
            interpreter: The interpreter used to encode the state
            #bilou_tagging: indicates whether BILOU tagging should be used or not

        Returns:
            A dictionary of entity type to list of features.
        """
        # TODO
        #  The entity states used to create the tag-idx-mapping contains the
        #  entities and the concatenated entity and roles/groups. We do not
        #  distinguish between entities and roles/groups right now.
        if (
            not entity_data
            or not self.encoding_specs
            or self.encoding_specs.num_tags < 2
        ):
            # we cannot build a classifier with fewer than 2 classes
            return {}

        message = interpreter.featurize_message(Message(entity_data))

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
