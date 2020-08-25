import logging
import numpy as np
import scipy.sparse
from typing import Tuple, List, Optional, Dict, Text
from collections import defaultdict

from rasa.utils import common as common_utils
from rasa.core.domain import Domain, STATE, SUB_STATE
from rasa.utils.features import Features
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.constants import USER, PREVIOUS_ACTION, FORM, SLOTS, ACTION
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

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    def __init__(self) -> None:

        super().__init__()
        self._default_feature_states = {}
        self.e2e_action_texts = []

    def prepare_from_domain(self, domain: Domain) -> None:
        # store feature states for each attribute in order to create binary features
        def convert_to_dict(feature_states: List[Text]) -> Dict[Text, int]:
            return {
                feature_state: idx for idx, feature_state in enumerate(feature_states)
            }

        if not domain.intents == []:
            self._default_feature_states[INTENT] = convert_to_dict(domain.intents)
        if not domain.action_names == []:
            self._default_feature_states[ACTION_NAME] = convert_to_dict(
                domain.action_names
            )
        if not domain.entities == []:
            self._default_feature_states[ENTITIES] = convert_to_dict(domain.entities)
        if not domain.slot_states == []:
            self._default_feature_states[SLOTS] = convert_to_dict(domain.slot_states)
        if not domain.form_names == []:
            self._default_feature_states[FORM] = convert_to_dict(domain.form_names)
        self.e2e_action_texts = domain.e2e_action_texts

    @staticmethod
    def _construct_message(
        sub_state: SUB_STATE, state_type: Text
    ) -> Tuple["Message", Text]:
        if state_type == USER:
            if sub_state.get(INTENT):
                message = Message(data={INTENT: sub_state.get(INTENT)})
                attribute = INTENT
            else:
                message = Message(data={TEXT: sub_state.get(TEXT)})
                attribute = TEXT
        elif state_type in {PREVIOUS_ACTION, ACTION}:
            if sub_state.get(ACTION_NAME):
                message = Message(data={ACTION_NAME: sub_state.get(ACTION_NAME)})
                attribute = ACTION_NAME
            else:
                message = Message(data={ACTION_TEXT: sub_state.get(ACTION_TEXT)})
                attribute = ACTION_TEXT
        else:
            raise ValueError(
                f"Given state_type '{state_type}' is not supported. "
                f"It must be either '{USER}' or '{PREVIOUS_ACTION}'."
            )

        return message, attribute

    def _create_features(
        self, sub_state: SUB_STATE, attribute: Text, sparse: bool = False
    ) -> Dict[Text, List["Features"]]:
        if attribute in {INTENT, ACTION_NAME}:
            state_features = {sub_state[attribute]: 1}
        elif attribute == ENTITIES:
            state_features = {entity: 1 for entity in sub_state.get(ENTITIES, [])}
        elif attribute == FORM:
            state_features = {sub_state["name"]: 1}
        elif attribute == SLOTS:
            state_features = {
                f"{slot_name}_{i}": value
                for slot_name, slot_as_feature in sub_state.items()
                for i, value in enumerate(slot_as_feature)
            }
        else:
            raise ValueError(
                f"Given attribute '{attribute}' is not supported. "
                f"It must be one of '{self._default_feature_states.keys()}'."
            )

        if attribute not in self._default_feature_states:
            return

        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            features[self._default_feature_states[attribute][state_feature]] = value
        features = np.expand_dims(features, 0)

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        features = Features(
            features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
        )
        return {attribute: [features]}

    def _extract_features(
        self,
        sub_state: SUB_STATE,
        state_type: Text,
        interpreter: Optional[NaturalLanguageInterpreter],
        sparse: bool = False,
    ) -> Dict[Text, List["Features"]]:

        output = defaultdict(list)
        message, attribute = self._construct_message(sub_state, state_type)

        if interpreter is not None:
            parsed_message = interpreter.synchronous_parse_message(message, attribute)
            all_features = (
                parsed_message.get_sparse_features(attribute)
                + parsed_message.get_dense_features(attribute)
                if parsed_message is not None
                else ()
            )

            for features in all_features:
                if features is not None:
                    output[attribute].append(features)

            for name_attribute in [INTENT, ACTION_NAME]:
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
                    output[name_attribute] += sentence_features

        output = dict(output)
        if not output.get(attribute) and attribute in {INTENT, ACTION_NAME}:
            # there can only be either TEXT or INTENT
            # or ACTION_TEXT or ACTION_NAME
            # therefore nlu pipeline didn't create features for user or action
            output = self._create_features(sub_state, attribute, sparse)

        return output

    def encode_state(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:

        featurized_state = {}
        for state_type, sub_state in state.items():
            if state_type in {USER, PREVIOUS_ACTION}:
                featurized_state.update(
                    self._extract_features(
                        sub_state, state_type, interpreter, sparse=True
                    )
                )
            if state_type == USER:
                if sub_state.get(ENTITIES):
                    featurized_state.update(
                        self._create_features(sub_state, ENTITIES, sparse=True)
                    )
            if state_type in {SLOTS, FORM}:
                if sub_state.get(state_type):
                    featurized_state.update(
                        self._create_features(sub_state, state_type, sparse=True)
                    )

        return featurized_state

    def _encode_action(
        self, action: Text, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:

        if action in self.e2e_action_texts:
            action_as_sub_state = {ACTION_TEXT: action}
        else:
            action_as_sub_state = {ACTION_NAME: action}

        return self._extract_features(action_as_sub_state, ACTION, interpreter)

    def create_encoded_all_actions(
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
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        # ignore nlu interpreter to create binary features
        return super().encode_state(state, None)

    def create_encoded_all_actions(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> List[Dict[Text, List["Features"]]]:
        # ignore nlu interpreter to create binary features
        return super().create_encoded_all_actions(domain, None)


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
