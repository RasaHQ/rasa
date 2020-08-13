import io
import jsonpickle
import logging
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Any, Union
import copy
from collections import deque
import scipy.sparse
from collections import defaultdict

import rasa.utils.io
from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain, STATE
from rasa.core.events import ActionExecuted, UserUttered, Form, SlotSet
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.data import DialogueTrainingData
from rasa.utils.common import is_logging_disabled
from rasa.utils.features import Features
from rasa.core.interpreter import (
    NaturalLanguageInterpreter,
    RegexInterpreter,
    RasaNLUInterpreter,
)
from rasa.core.constants import USER, PREVIOUS_ACTION, FORM, SLOTS, ACTION
from rasa.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    ENTITIES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURE_TYPE_SEQUENCE,
    FEATURE_TYPE_SENTENCE,
)
from rasa.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class SingleStateFeaturizer:
    """Base class for mechanisms to transform the conversations state into ML formats.

    Subclasses of SingleStateFeaturizer decide how the bot will transform
    the conversation state to a format which a classifier can read:
    feature vector.
    """

    def prepare_from_domain(self, domain: Domain) -> None:
        """Helper method to init based on domain."""

        pass

    def encode_state(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        """Encode user input."""

        raise NotImplementedError(
            "SingleStateFeaturizer must have "
            "the capacity to "
            "encode states to a feature vector"
        )

    def encode_action(
        self, action: Text, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> Dict[Text, List["Features"]]:
        """Encode user input."""

        raise NotImplementedError(
            "SingleStateFeaturizer must have "
            "the capacity to "
            "encode actions to a feature vector"
        )

    def create_encoded_all_actions(
        self, domain: Domain, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> List[Dict[Text, List["Features"]]]:
        """Create matrix with all actions from domain encoded in rows."""

        raise NotImplementedError("Featurizer must implement encoding actions.")


class BinarySingleStateFeaturizer(SingleStateFeaturizer):
    """Assumes all features are binary.

    All features should be either on or off, denoting them with 1 or 0.
    """

    def __init__(self) -> None:
        """Declares instant variables."""

        super().__init__()

        self.num_features = None
        self.input_state_map = None

    def prepare_from_domain(self, domain: Domain) -> None:
        """Use Domain to prepare featurizer."""

        self.num_features = domain.num_states
        self.input_state_map = domain.input_state_map

    def encode_state(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> np.ndarray:
        """Returns a binary vector indicating which features are active.

        Given a dictionary of states (e.g. 'intent_greet',
        'prev_action_listen',...) return a binary vector indicating which
        features of `self.input_features` are in the bag. NB it's a
        regular double precision float array type.

        For example with two active features out of five possible features
        this would return a vector like `[0 0 1 0 1]`

        If intent features are given with a probability, for example
        with two active features and two uncertain intents out
        of five possible features this would return a vector
        like `[0.3, 0.7, 1.0, 0, 1.0]`.

        If this is just a padding vector we set all values to `-1`.
        padding vectors are specified by a `None` or `[None]`
        value for states.
        """

        if not self.num_features:
            raise Exception(
                "BinarySingleStateFeaturizer was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            if state_name in self.input_state_map:
                idx = self.input_state_map[state_name]
                used_features[idx] = prob
                using_only_ints = using_only_ints and utils.is_int(prob)
            else:
                logger.debug(
                    "Feature '{}' (value: '{}') could not be found in "
                    "feature map. Make sure you added all intents and "
                    "entities to the domain".format(state_name, prob)
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows as bag of words"""

        return np.eye(domain.num_actions)


class LabelTokenizerSingleStateFeaturizer(SingleStateFeaturizer):
    """Creates bag-of-words feature vectors.

    User intents and bot action names are split into tokens
    and used to create bag-of-words feature vectors.

    Args:
        split_symbol: The symbol that separates words in
            intets and action names.

        use_shared_vocab: The flag that specifies if to create
            the same vocabulary for user intents and bot actions.
    """

    def __init__(
        self, use_shared_vocab: bool = False, split_symbol: Text = "_"
    ) -> None:
        """inits vocabulary for label bag of words representation"""
        super().__init__()

        self.use_shared_vocab = use_shared_vocab
        self.split_symbol = split_symbol

        self.num_features = None
        self.user_labels = []
        self.slot_labels = []
        self.bot_labels = []

        self.bot_vocab = None
        self.user_vocab = None

    @staticmethod
    def _create_label_token_dict(labels, split_symbol="_") -> Dict[Text, int]:
        """Splits labels into tokens by using provided symbol.

        Creates the lookup dictionary for this tokens.
        Values in this dict are used for featurization.
        """

        distinct_tokens = {
            token for label in labels for token in label.split(split_symbol)
        }
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def prepare_from_domain(self, domain: Domain) -> None:
        """Creates internal vocabularies for user intents and bot actions."""

        self.user_labels = domain.intents + domain.entities
        self.slot_labels = domain.slot_states + domain.form_names
        self.bot_labels = domain.action_names

        if self.use_shared_vocab:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels + self.user_labels, self.split_symbol
            )
            self.user_vocab = self.bot_vocab
        else:
            self.bot_vocab = self._create_label_token_dict(
                self.bot_labels, self.split_symbol
            )
            self.user_vocab = self._create_label_token_dict(
                self.user_labels, self.split_symbol
            )

        self.num_features = (
            len(self.user_vocab) + len(self.slot_labels) + len(self.bot_vocab)
        )

    def encode_state(
        self, state: STATE, interpreter: Optional[NaturalLanguageInterpreter]
    ) -> np.ndarray:
        """Returns a binary vector indicating which tokens are present."""

        if not self.num_features:
            raise Exception(
                "LabelTokenizerSingleStateFeaturizer "
                "was not prepared before encoding."
            )

        if state is None or None in state:
            return np.ones(self.num_features, dtype=np.int32) * -1

        # we are going to use floats and convert to int later if possible
        used_features = np.zeros(self.num_features, dtype=np.float)
        using_only_ints = True
        for state_name, prob in state.items():
            using_only_ints = using_only_ints and utils.is_int(prob)
            if state_name in self.user_labels:
                if PREV_PREFIX + ACTION_LISTEN_NAME in state:
                    # else we predict next action from bot action and memory
                    for t in state_name.split(self.split_symbol):
                        used_features[self.user_vocab[t]] += prob

            elif state_name in self.slot_labels:
                offset = len(self.user_vocab)
                idx = self.slot_labels.index(state_name)
                used_features[offset + idx] += prob

            elif state_name[len(PREV_PREFIX) :] in self.bot_labels:
                action_name = state_name[len(PREV_PREFIX) :]
                for t in action_name.split(self.split_symbol):
                    offset = len(self.user_vocab) + len(self.slot_labels)
                    idx = self.bot_vocab[t]
                    used_features[offset + idx] += prob

            else:
                logger.warning(
                    f"Feature '{state_name}' could not be found in feature map."
                )

        if using_only_ints:
            # this is an optimization - saves us a bit of memory
            return used_features.astype(np.int32)
        else:
            return used_features

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows as bag of words"""

        encoded_all_actions = np.zeros(
            (domain.num_actions, len(self.bot_vocab)), dtype=np.int32
        )
        for idx, name in enumerate(domain.action_names):
            for t in name.split(self.split_symbol):
                encoded_all_actions[idx, self.bot_vocab[t]] = 1
        return encoded_all_actions


class E2ESingleStateFeaturizer(SingleStateFeaturizer):
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

        # TODO consider using bool or int to save memory
        features = np.zeros(len(self._default_feature_states[attribute]), np.float32)
        for state_feature, value in state_features.items():
            features[self._default_feature_states[attribute][state_feature]] = value

        features = Features(
            features, FEATURE_TYPE_SENTENCE, attribute, self.__class__.__name__
        )
        return {attribute: [features]}

    def _extract_features(
        self,
        sub_state: Dict[Text, Union[Text, Tuple[float], Tuple[Text]]],
        state_type: Text,
        interpreter: NaturalLanguageInterpreter,
    ) -> Dict[Text, List["Features"]]:

        message, attribute = self._construct_message(sub_state, state_type)

        parsed_message = interpreter.synchronous_parse_message(message, attribute)
        all_features = (
            parsed_message.get_sparse_features(attribute)
            + parsed_message.get_dense_features(attribute)
            if parsed_message is not None
            else ()
        )

        output = defaultdict(list)
        for features in all_features:
            if features is not None:
                output[attribute].append(features)
        output = dict(output)

        if not output.get(attribute) and attribute in {INTENT, ACTION_NAME}:
            # there can only be either TEXT or INTENT
            # or ACTION_TEXT or ACTION_NAME
            # therefore nlu pipeline didn't create features for user or action
            output = self._create_features(sub_state, attribute)

        return output

    def encode_state(
        self, state: STATE, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:

        featurized_state = {}
        for state_type, sub_state in state.items():
            if state_type in {USER, PREVIOUS_ACTION}:
                featurized_state.update(
                    self._extract_features(sub_state, state_type, interpreter)
                )
            if state_type == USER:
                featurized_state.update(self._create_features(sub_state, ENTITIES))
            if state_type in {SLOTS, FORM}:
                featurized_state.update(self._create_features(sub_state, state_type))

        return featurized_state

    def encode_action(
        self, action: Text, interpreter: NaturalLanguageInterpreter
    ) -> Dict[Text, List["Features"]]:

        if action in self.e2e_action_texts:
            action_as_sub_state = {ACTION_TEXT: action}
        else:
            action_as_sub_state = {ACTION_NAME: action}

        return self._extract_features(action_as_sub_state, ACTION, interpreter)

    def create_encoded_all_actions(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> List[Dict[Text, List["Features"]]]:

        return [
            self.encode_action(action, interpreter) for action in domain.action_names
        ]


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

    def _featurize_labels(
        self,
        trackers_as_actions: List[List[Text]],
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        return [
            [
                self.state_featurizer.encode_action(action, interpreter)
                for action in tracker_actions
            ]
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
    ) -> DialogueTrainingData:
        """Create training data."""

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
        y = self._featurize_labels(trackers_as_actions, interpreter)

        return DialogueTrainingData(X, y)

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
