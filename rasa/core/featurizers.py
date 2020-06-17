import io
import jsonpickle
import logging
import numpy as np
import os
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Any
from scipy.sparse import csr_matrix
import re
import string

import rasa.utils.io
from rasa.core import utils
from rasa.core.actions.action import ACTION_LISTEN_NAME, default_action_names
from rasa.core.domain import PREV_PREFIX, Domain
from rasa.core.events import ActionExecuted, UserUttered, Event
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.data import DialogueTrainingData
import scipy.sparse

from rasa.core.interpreter import RasaE2EInterpreter
from rasa.nlu.constants import (
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TEXT,
)
from rasa.nlu.training_data import Message
from rasa.utils.common import is_logging_disabled
from rasa.utils import train_utils

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

    def encode(self, state: Dict[Text, float]) -> np.ndarray:
        """Encode user input."""

        raise NotImplementedError(
            "SingleStateFeaturizer must have "
            "the capacity to "
            "encode states to a feature vector"
        )

    @staticmethod
    def action_as_one_hot(action: Text, domain: Domain) -> np.ndarray:
        """Encode system action as one-hot vector."""

        if action is None:
            return np.ones(domain.num_actions, dtype=int) * -1

        y = np.zeros(domain.num_actions, dtype=int)
        y[domain.index_for_action(action)] = 1
        return y

    @staticmethod
    def action_as_index(action: Text, domain: Domain) -> np.ndarray:
        """Encode system action as one-hot vector."""

        if action is None:
            return np.ones(domain.num_actions, dtype=int) * -1

        y = domain.index_for_action(action.text)
        return y

    def create_encoded_all_actions(self, domain: Domain) -> np.ndarray:
        """Create matrix with all actions from domain encoded in rows."""

        raise NotImplementedError("Featurizer must implement encoding actions.")


class E2ESingleStateFeaturizer(SingleStateFeaturizer):
    def __init__(self) -> None:

        super().__init__()
        self.interpreter = RasaE2EInterpreter()

    def featurize_slots(self, slot_dict):
        slot_featurization = np.zeros(len(self.interpreter.slot_states))
        for slot_name in list(slot_dict.keys()):
            slot_featurization[self.interpreter.slot_states.index(slot_name)] = 1
        return None, slot_featurization

    def _extract_features(
        self, message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[np.ndarray]]:
        sparse_features = None
        dense_features = None

        #check that it is slot dictionary and not the processed empty used utterance
        if isinstance(message, Dict) and not 'intent' in message.keys():
            return self.featurize_slots(message)

        if message.get(SPARSE_FEATURE_NAMES[attribute]) is not None:
            sparse_features = message.get(SPARSE_FEATURE_NAMES[attribute])

        if message.get(DENSE_FEATURE_NAMES[attribute]) is not None:
            dense_features = message.get(DENSE_FEATURE_NAMES[attribute])

        if sparse_features is not None and dense_features is not None:
            if sparse_features.shape[0] != dense_features.shape[0]:
                raise ValueError(
                    f"Sequence dimensions for sparse and dense features "
                    f"don't coincide in '{message.text}' for attribute '{attribute}'."
                )

        sparse_features = train_utils.sequence_to_sentence_features(sparse_features)
        dense_features = train_utils.sequence_to_sentence_features(dense_features)

        return sparse_features, dense_features

    def combine_state_features(self, state_features):
        sparse_state, dense_state = None, None
        if (
            state_features["user"][0] is not None
            and state_features["prev_action"][0] is not None
        ):
            sparse_state = scipy.sparse.hstack(
                [state_features["user"][0], state_features["prev_action"][0]]
            )
        if (
            state_features["user"][1] is not None
            and state_features["prev_action"][1] is not None
        ):
            dense_state = np.hstack(
                (state_features["user"][1], state_features["prev_action"][1])
            )
        return sparse_state, dense_state

    def encode_e2e(
        self, state: Dict[Text, Event],
    ):
        """
        Encode the state into a numpy array or a sparse sklearn

        Args:
            - state: dictionary describing current state, state represented as a dicitonary {text: Event}, where Event is UserUttered/ActionExecuted
            - type output: type to return the features as (numpyarray or sklearn coo_matrix)
        Returns:
            - nparray(vocab_size,) or coo_matrix(1, vocab_size)
        """
        if not list(state.keys()) == []:
            state_extracted_features = {
                key: self._extract_features(state[key], TEXT) for key in state.keys()
            }
            if not "user" in state_extracted_features.keys():
                state_extracted_features["user"] = self._extract_features(
                    self.interpreter.interpreter.parse(
                        " ", only_output_properties=False
                    ),
                    TEXT,
                )

        sparse_state, dense_state = self.combine_state_features(
            state_extracted_features
        )

        slot_features = state_extracted_features['slots'][1]

        if self.interpreter.entities == []:
            entity_features = None
        else:
            entity_features = np.zeros(len(self.interpreter.entities))
            if "user" in list(state.keys()):
                if not state["user"].get("entities") is None:
                    user_entities = [
                        entity["entity"] for entity in state["user"].get("entities")
                    ]
                    for entity_name in user_entities:
                        entity_features[
                            self.interpreter.entities.index(entity_name)
                        ] = 1
        if entity_features is None:
            entity_slot_features = slot_features
        else:
            entity_slot_features = np.hstack((entity_features, slot_features))

        return sparse_state, dense_state, entity_slot_features

    def create_encoded_all_actions(self, domain):
        label_data = [
            (j, self._extract_features(self.interpreter.parse(action), TEXT))
            for j, action in enumerate(domain.action_names)
        ]
        return label_data


class TrackerFeaturizer:
    """Base class for actual tracker featurizers."""

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        use_intent_probabilities: bool = False,
    ) -> None:

        self.state_featurizer = state_featurizer
        self.use_intent_probabilities = use_intent_probabilities

    def _create_states(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        is_binary_training: bool = False,
    ) -> List[Dict[Text, float]]:
        """Create states: a list of dictionaries.

        If use_intent_probabilities is False (default behaviour),
        pick the most probable intent out of all provided ones and
        set its probability to 1.0, while all the others to 0.0.
        """

        states = tracker.past_states(domain)

        # during training we encounter only 1 or 0
        if not self.use_intent_probabilities and not is_binary_training:
            bin_states = []
            for state in states:
                # copy state dict to preserve internal order of keys
                bin_state = dict(state)
                best_intent = None
                best_intent_prob = -1.0
                for state_name, prob in state:
                    if state_name.startswith("intent_"):
                        if prob > best_intent_prob:
                            # finding the maximum confidence intent
                            if best_intent is not None:
                                # delete previous best intent
                                del bin_state[best_intent]
                            best_intent = state_name
                            best_intent_prob = prob
                        else:
                            # delete other intents
                            del bin_state[state_name]

                if best_intent is not None:
                    # set the confidence of best intent to 1.0
                    bin_state[best_intent] = 1.0

                bin_states.append(bin_state)
            return bin_states
        else:
            return [dict(state) for state in states]

    def collect_slots(self, tracker):
        current_nonnone_slots = {}
        for key, slot in tracker.slots.items():
            if slot is not None:
                for i, slot_value in enumerate(slot.as_feature()):
                    if slot_value != 0:
                        slot_id = f"slot_{key}_{i}"
                        current_nonnone_slots[slot_id] = slot_value
        return current_nonnone_slots

    def _create_states_e2e(self, tracker):
        import copy

        prev_tracker = None
        states = []
        for tr in tracker.generate_all_prior_trackers():
            if prev_tracker:
                state = tr.applied_events()[len(prev_tracker.applied_events()) :]
                state_dict = {}
                for event in state:
                    if isinstance(event, UserUttered):
                        if not event.message is None:
                            state_dict["user"] = event.message
                    elif isinstance(event, ActionExecuted):
                        if event.message is not None:
                            state_dict["prev_action"] = event.message
                        # to turn the default actions such as action_listen into Message;
                        else:
                            state_dict["prev_action"] = Message(event.action_name)
                    state_dict["slots"] = self.collect_slots(tr)
            else:
                state_dict = {}
            states.append(state_dict)
            prev_tracker = copy.deepcopy(tr)
        return states

    def _pad_states(self, states: List[Any]) -> List[Any]:
        """Pads states."""

        return states

    def _featurize_states(
        self, trackers_as_states: List[List[Dict[Text, float]]],
    ) -> Tuple[np.ndarray, List[int]]:
        """Create X."""

        features = []
        true_lengths = []

        for tracker_states in trackers_as_states:

            # len(trackers_as_states) = 1 means
            # it is called during prediction or we have
            # only one story, so no padding is needed

            if len(trackers_as_states) > 1:
                tracker_states = self._pad_states(tracker_states)

            story_features = [
                self.state_featurizer.encode_e2e(state)
                for state in tracker_states
                if not state is None and not state == {}
            ]

            dialogue_len = len(story_features)

            if not story_features == []:
                features.append(np.array(story_features))
                true_lengths.append(dialogue_len)

        # noinspection PyPep8Naming
        X = np.array(features)

        return X, true_lengths

    def _featurize_labels(
        self, trackers_as_actions: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Create y."""

        labels = []
        for tracker_actions in trackers_as_actions:

            if len(trackers_as_actions) > 1:
                tracker_actions = self._pad_states(tracker_actions)

            story_labels = [
                self.state_featurizer.action_as_index(action, domain)
                for action in tracker_actions
            ]
            for action in tracker_actions:
                sparse, dense = self.state_featurizer._extract_features(action, TEXT)
                value = (sparse.tocsr(), dense)

            labels.append(story_labels)

        y = np.array(labels)
        if y.ndim == 3 and isinstance(self, MaxHistoryTrackerFeaturizer):
            # if it is MaxHistoryFeaturizer, remove time axis
            y = y[:, 0, :]

        return y

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Dict]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions."""

        raise NotImplementedError(
            "Featurizer must have the capacity to encode trackers to feature vectors"
        )

    def featurize_trackers(
        self, trackers: List[DialogueStateTracker], domain: Domain, **kwargs
    ) -> DialogueTrainingData:
        """Create training data."""

        if self.state_featurizer is None:
            raise ValueError(
                "Variable 'state_featurizer' is not set. Provide "
                "'SingleStateFeaturizer' class to featurize trackers."
            )

        (
            trackers_as_states,
            trackers_as_actions,
        ) = self.training_states_and_actions_e2e(trackers, domain)

        self.state_featurizer.interpreter.prepare_training_data_and_train(
            trackers_as_states, trackers_as_actions, kwargs["output_path_nlu"], domain
        )

        # noinspection PyPep8Naming
        X, true_lengths = self._featurize_states(trackers_as_states)
        y = self._featurize_labels(trackers_as_actions, domain)

        return DialogueTrainingData(X, y, true_lengths)

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:
        """Transforms list of trackers to lists of states for prediction."""

        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )

    # noinspection PyPep8Naming
    def create_X(
        self, trackers: List[DialogueStateTracker], domain: Domain,
    ) -> np.ndarray:
        """Create X for prediction."""

        trackers_as_states = self.prediction_states(trackers, domain)
        X, _ = self._featurize_states(trackers_as_states)
        return X

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

    def __init__(
        self,
        state_featurizer: SingleStateFeaturizer,
        use_intent_probabilities: bool = False,
    ) -> None:

        super().__init__(state_featurizer, use_intent_probabilities)
        self.max_len = None

    @staticmethod
    def _calculate_max_len(trackers_as_actions) -> Optional[int]:
        """Calculate the length of the longest dialogue."""

        if trackers_as_actions:
            return max([len(states) for states in trackers_as_actions])
        else:
            return None

    def _pad_states(self, states: List[Any]) -> List[Any]:
        """Pads states up to max_len."""

        if len(states) < self.max_len:
            states += [None] * (self.max_len - len(states))

        return states

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Dict]], List[List[Text]]]:
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
            states = self._create_states(tracker, domain, is_binary_training=True)

            delete_first_state = False
            actions = []
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    if not event.unpredictable:
                        # only actions which can be
                        # predicted at a stories start
                        actions.append(event.action_name)
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

        self.max_len = self._calculate_max_len(trackers_as_actions)
        logger.debug(f"The longest dialogue has {self.max_len} actions.")

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:
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

    MAX_HISTORY_DEFAULT = None

    def __init__(
        self,
        state_featurizer: Optional[SingleStateFeaturizer] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
        use_intent_probabilities: bool = False,
    ) -> None:

        super().__init__(state_featurizer, use_intent_probabilities)
        self.max_history = max_history or self.MAX_HISTORY_DEFAULT
        self.remove_duplicates = remove_duplicates

    @staticmethod
    def slice_state_history(
        states: List[Dict[Text, float]], slice_length: int
    ) -> List[Optional[Dict[Text, float]]]:
        """Slices states from the trackers history.

        If the slice is at the array borders, padding will be added to ensure
        the slice length.
        """

        slice_end = len(states)
        if slice_length == None:
            slice_start = 0
        else:
            slice_start = max(0, slice_end - slice_length)
        # noinspection PyTypeChecker
        state_features = states[slice_start:]
        return state_features

    @staticmethod
    def _hash_example(states, action) -> int:
        """Hash states for efficient deduplication."""
        states = [
            {key: value.text for key, value in s.items() if not key == "slots"}
            for s in states
        ]
        frozen_states = tuple(s if s is None else frozenset(s.items()) for s in states)
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def training_states_and_actions_e2e(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Optional[Dict[Text, float]]]], List[List[Text]]]:
        trackers_as_states_e2e = []
        trackers_as_actions_e2e = []
        hashed_examples = set()
        pbar = tqdm(
            trackers, desc="Processed trackers e2e", disable=is_logging_disabled()
        )
        for tracker in pbar:
            states = self._create_states_e2e(tracker)
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
                                sliced_states, event.action_name
                            )

                            # only continue with tracker_states that created a
                            # hashed_featurization we haven't observed
                            if hashed not in hashed_examples:
                                hashed_examples.add(hashed)
                                trackers_as_states_e2e.append(sliced_states)

                                if len(sliced_states) > 1:
                                    if event.message is not None:
                                        trackers_as_actions_e2e.append(
                                            [Message(event.message.text)]
                                        )
                                    # if it is a default action, turn it into a message
                                    else:
                                        trackers_as_actions_e2e.append(
                                            [Message(event.action_name)]
                                        )
                        else:
                            trackers_as_states_e2e.append(sliced_states)
                            if len(sliced_states) > 1:
                                if event.message is not None:
                                    trackers_as_actions_e2e.append(
                                        [Message(event.message.text)]
                                    )
                                # if it is a default action, turn it into a message
                                else:
                                    trackers_as_actions_e2e.append(
                                        [Message(event.action_name)]
                                    )
                        pbar.set_postfix(
                            {"# actions": "{:d}".format(len(trackers_as_actions_e2e))}
                        )
                    idx += 1

        return trackers_as_states_e2e, trackers_as_actions_e2e

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[Optional[Dict[Text, float]]]], List[List[Text]]]:
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
            states = self._create_states(tracker, domain, True)
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
                                sliced_states, event.action_name
                            )

                            # only continue with tracker_states that created a
                            # hashed_featurization we haven't observed
                            if hashed not in hashed_examples:
                                hashed_examples.add(hashed)
                                trackers_as_states.append(sliced_states)
                                trackers_as_actions.append([event.action_name])
                        else:
                            trackers_as_states.append(sliced_states)
                            trackers_as_actions.append([event.action_name])

                        pbar.set_postfix(
                            {"# actions": "{:d}".format(len(trackers_as_actions))}
                        )
                    idx += 1

            logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[Dict[Text, float]]]:
        from rasa.nlu.model import Interpreter

        """Transforms list of trackers to lists of states for prediction."""
        trackers_as_states = [self._create_states_e2e(tracker) for tracker in trackers]
        # !!! To have DIET at prediction time, we would need to load it at 
        # every prediction step which I don't think is a good idea.  

        trackers_as_states_modified = []
        for tracker in trackers_as_states:
            curr_tracker = []
            for state in tracker:
                curr_state = {}
                for key, value in state.items():
                    if isinstance(value, Message):
                        curr_state[key] = self.state_featurizer.interpreter.parse(
                            value.text
                        )
                        curr_state[key]["entities"] = value.get("entities")
                    else:
                        curr_state[key] = value
                curr_tracker.append(curr_state)
            trackers_as_states_modified.append(curr_tracker)

        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states_modified
        ]

        return trackers_as_states
