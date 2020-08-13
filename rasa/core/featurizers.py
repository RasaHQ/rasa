import jsonpickle
import logging
import numpy as np
import scipy.sparse
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Union
from collections import deque
from collections import defaultdict

import rasa.utils.io
from rasa.utils import common as common_utils
from rasa.core.domain import Domain, STATE
from rasa.core.events import ActionExecuted
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled
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
        self._default_feature_states[INTENT] = {
            f: i for i, f in enumerate(domain.intents)
        }
        self._default_feature_states[ACTION_NAME] = {
            f: i for i, f in enumerate(domain.action_names)
        }
        self._default_feature_states[ENTITIES] = {
            f: i for i, f in enumerate(domain.entities)
        }
        self._default_feature_states[SLOTS] = {
            f: i for i, f in enumerate(domain.slot_states)
        }
        self._default_feature_states[FORM] = {
            f: i for i, f in enumerate(domain.form_names)
        }
        self.e2e_action_texts = domain.e2e_action_texts

    @staticmethod
    def _construct_message(
        sub_state: Dict[Text, Union[Text, Tuple[float], Tuple[Text]]], state_type: Text
    ) -> Tuple["Message", Text]:
        if state_type == USER:
            if sub_state.get(INTENT):
                message = Message(data={INTENT: sub_state.get(INTENT)})
                attribute = INTENT
            else:
                message = Message(sub_state.get(TEXT))
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
        self,
        sub_state: Dict[Text, Union[Text, Tuple[float], Tuple[Text]]],
        attribute: Text,
        sparse: bool = False,
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

        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            features[self._default_feature_states[attribute][state_feature]] = value

        if sparse:
            features = scipy.sparse.coo_matrix(features)

        features = Features(
            features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
        )
        return {attribute: [features]}

    def _extract_features(
        self,
        sub_state: Dict[Text, Union[Text, Tuple[float], Tuple[Text]]],
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
                featurized_state.update(
                    self._create_features(sub_state, ENTITIES, sparse=True)
                )
            if state_type in {SLOTS, FORM}:
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


class TrackerFeaturizer:
    """Base class for actual tracker featurizers."""

    def __init__(
        self, state_featurizer: Optional[SingleStateFeaturizer] = None
    ) -> None:

        self.state_featurizer = state_featurizer

    @staticmethod
    def _unfreeze_states(states: deque) -> List[STATE]:
        return [
            {key: dict(value) for key, value in dict(state).items()} for state in states
        ]

    def _create_states(
        self, tracker: DialogueStateTracker, domain: Domain,
    ) -> List[STATE]:
        """Create states: a list of dictionaries.

        If use_intent_probabilities is False (default behaviour),
        pick the most probable intent out of all provided ones and
        set its probability to 1.0, while all the others to 0.0.
        """

        states = tracker.past_states(domain)

        return self._unfreeze_states(states)

    def _featurize_states(
        self,
        trackers_as_states: List[List[STATE]],
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        return [
            [
                self.state_featurizer.encode_state(state, interpreter)
                for state in tracker_states
            ]
            for tracker_states in trackers_as_states
        ]

    @staticmethod
    def _convert_labels_to_ids(
        trackers_as_actions: List[List[Text]], domain: Domain,
    ) -> List[List[int]]:
        return [
            [domain.index_for_action(action) for action in tracker_actions]
            for tracker_actions in trackers_as_actions
        ]

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions."""

        raise NotImplementedError(
            "Featurizer must have the capacity to encode trackers to feature vectors"
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[
        List[List[Dict[Text, List["Features"]]]], List[List[int]],
    ]:
        if self.state_featurizer is None:
            raise ValueError(
                "Variable 'state_featurizer' is not set. Provide "
                "'SingleStateFeaturizer' class to featurize trackers."
            )

        self.state_featurizer.prepare_from_domain(domain)

        trackers_as_states, trackers_as_actions = self.training_states_and_actions(
            trackers, domain
        )

        # noinspection PyPep8Naming
        X = self._featurize_states(trackers_as_states, interpreter)
        label_ids = self._convert_labels_to_ids(trackers_as_actions, domain)

        return X, label_ids

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[STATE]]:
        """Transforms list of trackers to lists of states for prediction."""

        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )

    # noinspection PyPep8Naming
    def create_X(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        """Create X for prediction."""

        trackers_as_states = self.prediction_states(trackers, domain)
        return self._featurize_states(trackers_as_states, interpreter)

    def persist(self, path) -> None:
        featurizer_file = os.path.join(path, "featurizer.json")
        rasa.utils.io.create_directory_for_file(featurizer_file)

        # noinspection PyTypeChecker
        rasa.utils.io.write_text_file(str(jsonpickle.encode(self)), featurizer_file)

    @staticmethod
    def load(path) -> Optional["TrackerFeaturizer"]:
        """Loads the featurizer from file."""

        featurizer_file = os.path.join(path, "featurizer.json")
        if os.path.isfile(featurizer_file):
            return jsonpickle.decode(rasa.utils.io.read_file(featurizer_file))
        else:
            logger.error(
                "Couldn't load featurizer for policy. "
                "File '{}' doesn't exist.".format(featurizer_file)
            )
            return None


class FullDialogueTrackerFeaturizer(TrackerFeaturizer):
    """Creates full dialogue training data for time distributed architectures.

    Creates training data that uses each time output for prediction.
    Training data is padded up to the length of the longest dialogue with -1.
    """

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Training data is padded up to the length of the longest dialogue with -1.
        """

        trackers_as_states = []
        trackers_as_actions = []

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            delete_first_state = False
            actions = []
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        actions.append(event.action_name or event.e2e_text)
                    else:
                        # unpredictable actions can be
                        # only the first in the story
                        if delete_first_state:
                            raise Exception(
                                "Found two unpredictable "
                                "actions in one story."
                                "Check your story files."
                            )
                        else:
                            delete_first_state = True

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[STATE]]:
        """Transforms list of trackers to lists of states for prediction."""

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]

        return trackers_as_states


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):
    """Slices the tracker history into max_history batches.

    Creates training data that uses last output for prediction.
    Training data is padded up to the max_history with -1.
    """

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
    ) -> None:

        super().__init__(state_featurizer)
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates

    @staticmethod
    def slice_state_history(
        states: List[STATE], slice_length: Optional[int]
    ) -> List[STATE]:
        """Slices states from the trackers history.
        If the slice is at the array borders, padding will be added to ensure
        the slice length.
        """
        if not slice_length:
            return states

        return states[-slice_length:]

    @staticmethod
    def _hash_example(
        states: List[STATE], action: Text, tracker: DialogueStateTracker,
    ) -> int:
        """Hash states for efficient deduplication."""
        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[STATE]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.
        Training data is padded up to the max_history with -1.
        """

        trackers_as_states = []
        trackers_as_actions = []

        # from multiple states that create equal featurizations
        # we only need to keep one.
        hashed_examples = set()

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(trackers, desc="Processed trackers", disable=is_logging_disabled())
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            idx = 0
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        sliced_states = self.slice_state_history(
                            states[: idx + 1], self.max_history
                        )
                        if self.remove_duplicates:
                            hashed = self._hash_example(
                                sliced_states,
                                event.action_name or event.e2e_text,
                                tracker,
                            )

                            # only continue with tracker_states that created a
                            # hashed_featurization we haven't observed
                            if hashed not in hashed_examples:
                                hashed_examples.add(hashed)
                                trackers_as_states.append(sliced_states)
                                trackers_as_actions.append(
                                    [event.action_name or event.e2e_text]
                                )
                        else:
                            trackers_as_states.append(sliced_states)
                            trackers_as_actions.append(
                                [event.action_name or event.e2e_text]
                            )

                        pbar.set_postfix(
                            {"# actions": "{:d}".format(len(trackers_as_actions))}
                        )
                    idx += 1

        logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[STATE]]:
        """Transforms list of trackers to lists of states for prediction."""

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]

        return trackers_as_states
