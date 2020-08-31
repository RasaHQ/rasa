import logging
import numpy as np
import scipy.sparse
from typing import Tuple, List, Optional, Dict, Text
from collections import defaultdict

from rasa.utils import common as common_utils
from rasa.core.domain import Domain, State, SubState
from rasa.utils.features import Features
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.constants import USER, PREVIOUS_ACTION, SLOTS, ACTION, ACTIVE_LOOP
from rasa.constants import DOCS_URL_MIGRATION_GUIDE
from rasa.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    FEATURE_TYPE_SENTENCE,
)
from rasa.nlu.training_data.message import Message
from rasa.core.trackers import prev_action_listen_in_state

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    def __init__(self) -> None:
        self._default_feature_states = {}
        self.e2e_action_texts = []

    def prepare_from_domain(self, domain: Domain) -> None:
        # store feature states for each attribute in order to create binary features
        def convert_to_dict(feature_states: List[Text]) -> Dict[Text, int]:
            return {
                feature_state: idx for idx, feature_state in enumerate(feature_states)
            }

        self._default_feature_states[INTENT] = convert_to_dict(domain.intents)
        self._default_feature_states[ACTION_NAME] = convert_to_dict(domain.action_names)
        self._default_feature_states[ENTITIES] = convert_to_dict(domain.entities)
        self._default_feature_states[SLOTS] = convert_to_dict(domain.slot_states)
        self._default_feature_states[ACTIVE_LOOP] = convert_to_dict(domain.form_names)
        self.e2e_action_texts = domain.e2e_action_texts

    def _state_features_for_attribute(
        self, sub_state: SubState, attribute: Text
    ) -> Dict[Text, int]:
        if attribute in {INTENT, ACTION_NAME}:
            return {sub_state[attribute]: 1}
        elif attribute == ENTITIES:
            return {entity: 1 for entity in sub_state.get(ENTITIES, [])}
        elif attribute == ACTIVE_LOOP:
            return {sub_state["name"]: 1}
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

    def _create_features(
        self, sub_state: SubState, attribute: Text, sparse: bool = False
    ) -> Dict[Text, List["Features"]]:
        state_features = self._state_features_for_attribute(sub_state, attribute)

        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            # check that the value is in default_feature_states to be able to assigh its value
            if state_feature in self._default_feature_states[attribute]:
                features[self._default_feature_states[attribute][state_feature]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        features = Features(
            features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
        )
        return {attribute: [features]}

    def _extract_state_features(
        self,
        sub_state: SubState,
        state_type: Text,
        interpreter: Optional[NaturalLanguageInterpreter],
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:

        output = defaultdict(list)
        message = Message(data=sub_state)
        # remove entities from possible attributes
        attributes = set(
            attribute for attribute in sub_state.keys() if attribute != ENTITIES
        )

        if interpreter is not None:
            parsed_message = interpreter.synchronous_parse_message(message)
            for attribute in attributes:
                if parsed_message is not None:
                    all_features = parsed_message.get_sparse_features(
                        attribute
                    ) + parsed_message.get_dense_features(attribute)
                else:
                    all_features = ()

                for features in all_features:
                    if features is not None:
                        output[attribute].append(features)

            # transform sequence sparse features to sentence sparse features
            # for intent and action_name
            for name_attribute in {INTENT, ACTION_NAME}:
                if output.get(name_attribute):
                    sentence_features = []
                    for feature in output.get(name_attribute):
                        sentence_features.append(
                            Features(
                                scipy.sparse.coo_matrix(feature.features.sum(0)),
                                FEATURE_TYPE_SENTENCE,
                                name_attribute,
                                feature.origin,
                            )
                        )
                    output[name_attribute] = sentence_features

        name_attribute = next(
            (
                attribute
                for attribute in attributes
                if attribute in {INTENT, ACTION_NAME}
            ),
            None,
        )
        if not output and name_attribute:
            # nlu pipeline didn't create features for user or action
            output = self._create_features(sub_state, name_attribute, sparse)

        return output

    def encode_state(
        self, state: State, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        state_features = {}
        for state_type, sub_state in state.items():
            if state_type == PREVIOUS_ACTION:
                state_features.update(
                    self._extract_state_features(
                        sub_state, state_type, interpreter, sparse=True
                    )
                )
            # featurize user only if it is "real" user input,
            # i.e. input from a turn after action_listen
            if state_type == USER and prev_action_listen_in_state(state):
                state_features.update(
                    self._extract_state_features(
                        sub_state, state_type, interpreter, sparse=True
                    )
                )
                if sub_state.get(ENTITIES):
                    state_features.update(
                        self._create_features(sub_state, ENTITIES, sparse=True)
                    )
            if state_type in {SLOTS, ACTIVE_LOOP}:
                state_features.update(
                    self._create_features(sub_state, state_type, sparse=True)
                )

        return state_features

    def _encode_action(
        self, action: Text, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:

        if action in self.e2e_action_texts:
            action_as_sub_state = {ACTION_TEXT: action}
        else:
            action_as_sub_state = {ACTION_NAME: action}

        return self._extract_state_features(action_as_sub_state, ACTION, interpreter)

    def encode_all_actions(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> List[Dict[Text, List["Features"]]]:

        return [
            self._encode_action(action, interpreter) for action in domain.action_names
        ]


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self) -> None:
        super().__init__()
        common_utils.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            category=FutureWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )

    def encode_state(
        self, state: State, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        # ignore nlu interpreter to create binary features
        return super().encode_state(state, None)

    def encode_all_actions(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> List[Dict[Text, List["Features"]]]:
        # ignore nlu interpreter to create binary features
        return super().encode_all_actions(domain, None)


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # it is hard to fully mimic old behavior, but SingleStateFeaturizer
        # does the same thing if nlu pipeline is configured correctly
        common_utils.raise_warning(
            f"'{self.__class__.__name__}' is deprecated and "
            f"will be removed in the future. "
            f"It is recommended to use the '{SingleStateFeaturizer.__name__}' instead.",
            category=FutureWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )
