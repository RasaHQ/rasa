from pathlib import Path

import jsonpickle
import logging

from rasa.shared.exceptions import RasaException
from rasa.shared.nlu.constants import TEXT
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Union
import numpy as np

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.core.constants import USER
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.features import Features

FEATURIZER_FILE = "featurizer.json"

logger = logging.getLogger(__name__)


class InvalidStory(RasaException):
    """Exception that can be raised if story cannot be featurized."""

    def __init__(self, message) -> None:
        self.message = message
        super(InvalidStory, self).__init__()

    def __str__(self) -> Text:
        return self.message


class TrackerFeaturizer:
    """Base class for actual tracker featurizers."""

    def __init__(
        self, state_featurizer: Optional[SingleStateFeaturizer] = None
    ) -> None:
        """Initialize the tracker featurizer.

        Args:
            state_featurizer: The state featurizer used to encode the states.
        """
        self.state_featurizer = state_featurizer

    @staticmethod
    def _create_states(tracker: DialogueStateTracker, domain: Domain) -> List[State]:
        """Create states for the given tracker.

        Args:
            tracker: a :class:`rasa.core.trackers.DialogueStateTracker`
            domain: a :class:`rasa.shared.core.domain.Domain`

        Returns:
            a list of states
        """
        return tracker.past_states(domain)

    def _featurize_states(
        self,
        trackers_as_states: List[List[State]],
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
        trackers_as_actions: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        # store labels in numpy arrays so that it corresponds to np arrays of input features
        return np.array(
            [
                np.array(
                    [domain.index_for_action(action) for action in tracker_actions]
                )
                for tracker_actions in trackers_as_actions
            ]
        )

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states and list of actions.
        """
        raise NotImplementedError(
            "Featurizer must have the capacity to encode trackers to feature vectors"
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[List[List[Dict[Text, List["Features"]]]], np.ndarray]:
        """Featurize the training trackers.

        Args:
            trackers: list of training trackers
            domain: the domain
            interpreter: the interpreter

        Returns:
            - a dictionary of state types (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
              ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
              turns in all training trackers
            - the label ids (e.g. action ids) for every dialuge turn in all training
              trackers
        """
        if self.state_featurizer is None:
            raise ValueError(
                f"Instance variable 'state_featurizer' is not set. "
                f"During initialization set 'state_featurizer' to an instance of "
                f"'{SingleStateFeaturizer.__class__.__name__}' class "
                f"to get numerical features for trackers."
            )

        self.state_featurizer.prepare_from_domain(domain)

        trackers_as_states, trackers_as_actions = self.training_states_and_actions(
            trackers, domain
        )

        tracker_state_features = self._featurize_states(trackers_as_states, interpreter)
        label_ids = self._convert_labels_to_ids(trackers_as_actions, domain)

        return tracker_state_features, label_ids

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A list of states.
        """
        raise NotImplementedError(
            "Featurizer must have the capacity to create feature vector"
        )

    def create_state_features(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        """Create state features for prediction.

        Args:
            trackers: A list of state trackers
            domain: The domain
            interpreter: The interpreter

        Returns:
            A dictionary of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        trackers_as_states = self.prediction_states(trackers, domain)
        return self._featurize_states(trackers_as_states, interpreter)

    def persist(self, path: Union[Text, Path]) -> None:
        """Persist the tracker featurizer to the given path.

        Args:
            path: The path to persist the tracker featurizer to.
        """
        featurizer_file = Path(path) / FEATURIZER_FILE
        rasa.shared.utils.io.create_directory_for_file(featurizer_file)

        # noinspection PyTypeChecker
        rasa.shared.utils.io.write_text_file(
            str(jsonpickle.encode(self)), featurizer_file
        )

    @staticmethod
    def load(path: Text) -> Optional["TrackerFeaturizer"]:
        """Load the featurizer from file.

        Args:
            path: The path to load the tracker featurizer from.

        Returns:
            The loaded tracker featurizer.
        """
        featurizer_file = Path(path) / FEATURIZER_FILE
        if featurizer_file.is_file():
            return jsonpickle.decode(rasa.shared.utils.io.read_file(featurizer_file))

        logger.error(
            f"Couldn't load featurizer for policy. "
            f"File '{featurizer_file}' doesn't exist."
        )
        return None


class FullDialogueTrackerFeaturizer(TrackerFeaturizer):
    """Creates full dialogue training data for time distributed architectures.

    Creates training data that uses each time output for prediction.
    Training data is padded up to the length of the longest dialogue with -1.
    """

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Training data is padded up to the length of the longest dialogue with -1.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states and list of actions.
        """

        trackers_as_states = []
        trackers_as_actions = []

        logger.debug(
            "Creating states and action examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            delete_first_state = False
            actions = []
            for event in tracker.applied_events():
                if not isinstance(event, ActionExecuted):
                    continue

                if not event.unpredictable:
                    # only actions which can be
                    # predicted at a stories start
                    actions.append(event.action_name or event.action_text)
                else:
                    # unpredictable actions can be
                    # only the first in the story
                    if delete_first_state:
                        raise InvalidStory(
                            f"Found two unpredictable actions in one story "
                            f"'{tracker.sender_id}'. Check your story files."
                        )
                    delete_first_state = True

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A list of states.
        """

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        # TODO there is no prediction support for e2e input right now, therefore
        #  temporary remove TEXT features from USER state during prediction
        for states in trackers_as_states:
            for state in states:
                if state.get(USER, {}).get(TEXT):
                    del state[USER][TEXT]

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
        states: List[State], slice_length: Optional[int]
    ) -> List[State]:
        """Slice states from the trackers history.

        If the slice is at the array borders, padding will be added to ensure
        the slice length.

        Args:
            states: The states
            slice_length: The slice length

        Returns:
            The sliced states.
        """
        if not slice_length:
            return states

        return states[-slice_length:]

    @staticmethod
    def _hash_example(
        states: List[State], action: Text, tracker: DialogueStateTracker
    ) -> int:
        """Hash states for efficient deduplication."""
        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        frozen_actions = (action,)
        return hash((frozen_states, frozen_actions))

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Training data is padded up to the length of the longest dialogue with -1.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states and list of actions.
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
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:
            states = self._create_states(tracker, domain)

            states_length_for_action = 0
            for event in tracker.applied_events():
                if not isinstance(event, ActionExecuted):
                    continue

                states_length_for_action += 1

                # use only actions which can be predicted at a stories start
                if event.unpredictable:
                    continue

                sliced_states = self.slice_state_history(
                    states[:states_length_for_action], self.max_history
                )
                if self.remove_duplicates:
                    hashed = self._hash_example(
                        sliced_states, event.action_name or event.action_text, tracker
                    )

                    # only continue with tracker_states that created a
                    # hashed_featurization we haven't observed
                    if hashed not in hashed_examples:
                        hashed_examples.add(hashed)
                        trackers_as_states.append(sliced_states)
                        trackers_as_actions.append(
                            [event.action_name or event.action_text]
                        )
                else:
                    trackers_as_states.append(sliced_states)
                    trackers_as_actions.append([event.action_name or event.action_text])

                pbar.set_postfix({"# actions": "{:d}".format(len(trackers_as_actions))})

        logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions

    def prediction_states(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A list of states.
        """

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]
        # TODO there is no prediction support for e2e input right now, therefore
        #  temporary remove TEXT features from USER state during prediction
        for states in trackers_as_states:
            for state in states:
                if state.get(USER, {}).get(TEXT):
                    del state[USER][TEXT]

        return trackers_as_states
