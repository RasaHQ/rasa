from pathlib import Path

import jsonpickle
import logging

from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Union, Any
import numpy as np

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted, UserUttered
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    is_prev_action_listen_in_state,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.core.constants import USER
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.exceptions import RasaException
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
        # store labels in numpy arrays so that it corresponds to np arrays of input
        # features
        return np.array(
            [
                np.array(
                    [domain.index_for_action(action) for action in tracker_actions]
                )
                for tracker_actions in trackers_as_actions
            ]
        )

    def _create_entity_tags(
        self,
        trackers_as_entities: List[List[Dict[Text, Any]]],
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        return [
            [
                self.state_featurizer.encode_entities(entity_data, interpreter)
                for entity_data in trackers_entities
            ]
            for trackers_entities in trackers_as_entities
        ]

    @staticmethod
    def _entity_data(event: UserUttered) -> Dict[Text, Any]:
        # train stories support both text and intent,
        # but if intent is present, the text is ignored
        if event.text and not event.intent_name:
            return {TEXT: event.text, ENTITIES: event.entities}

        # input is not textual, so add empty dict
        return {}

    @staticmethod
    def _remove_user_text_if_intent(trackers_as_states: List[List[State]]) -> None:
        for states in trackers_as_states:
            for state in states:
                # remove text features to only use intent
                if state.get(USER, {}).get(INTENT) and state.get(USER, {}).get(TEXT):
                    del state[USER][TEXT]

    def training_states_actions_and_entities(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, actions and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states, list of actions and list of entity data.
        """
        raise NotImplementedError(
            f"`{self.__class__.__name__}` should implement how to encode trackers as feature vectors"
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
        (
            trackers_as_states,
            trackers_as_actions,
            _,
        ) = self.training_states_actions_and_entities(trackers, domain)
        return trackers_as_states, trackers_as_actions

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[
        List[List[Dict[Text, List["Features"]]]],
        np.ndarray,
        List[List[Dict[Text, List["Features"]]]],
    ]:
        """Featurize the training trackers.

        Args:
            trackers: list of training trackers
            domain: the domain
            interpreter: the interpreter

        Returns:
            - a dictionary of state types (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
              ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
              turns in all training trackers
            - the label ids (e.g. action ids) for every dialogue turn in all training
              trackers
            - A dictionary of entity type (ENTITY_TAGS) to a list of features
              containing entity tag ids for text user inputs otherwise empty dict
              for all dialogue turns in all training trackers
        """
        if self.state_featurizer is None:
            raise ValueError(
                f"Instance variable 'state_featurizer' is not set. "
                f"During initialization set 'state_featurizer' to an instance of "
                f"'{SingleStateFeaturizer.__class__.__name__}' class "
                f"to get numerical features for trackers."
            )

        self.state_featurizer.prepare_for_training(domain, interpreter)

        (
            trackers_as_states,
            trackers_as_actions,
            trackers_as_entities,
        ) = self.training_states_actions_and_entities(trackers, domain)

        tracker_state_features = self._featurize_states(trackers_as_states, interpreter)
        label_ids = self._convert_labels_to_ids(trackers_as_actions, domain)
        entity_tags = self._create_entity_tags(trackers_as_entities, interpreter)

        return tracker_state_features, label_ids, entity_tags

    def _choose_last_user_input(
        self, trackers_as_states: List[List[State]], use_text_for_last_user_input: bool
    ) -> None:
        for states in trackers_as_states:
            last_state = states[-1]
            # only update the state of the real user utterance
            if not is_prev_action_listen_in_state(last_state):
                continue

            if use_text_for_last_user_input:
                # remove intent features to only use text
                if last_state.get(USER, {}).get(INTENT):
                    del last_state[USER][INTENT]
                # don't add entities if text is used for featurization
                if last_state.get(USER, {}).get(ENTITIES):
                    del last_state[USER][ENTITIES]
            else:
                # remove text features to only use intent
                if last_state.get(USER, {}).get(TEXT):
                    del last_state[USER][TEXT]

        # make sure that all dialogue steps are either intent or text based
        self._remove_user_text_if_intent(trackers_as_states)

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

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
        use_text_for_last_user_input: bool = False,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        """Create state features for prediction.

        Args:
            trackers: A list of state trackers
            domain: The domain
            interpreter: The interpreter
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

        Returns:
            A dictionary of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        trackers_as_states = self.prediction_states(
            trackers, domain, use_text_for_last_user_input
        )
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

    def training_states_actions_and_entities(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, actions and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states, list of actions and list of entity data.
        """
        trackers_as_states = []
        trackers_as_actions = []
        trackers_as_entities = []

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
            entities = []
            entity_data = {}
            for event in tracker.applied_events():
                if isinstance(event, UserUttered):
                    entity_data = self._entity_data(event)

                if not isinstance(event, ActionExecuted):
                    continue

                if not event.unpredictable:
                    # only actions which can be
                    # predicted at a stories start
                    actions.append(event.action_name or event.action_text)
                    entities.append(entity_data)
                else:
                    # unpredictable actions can be
                    # only the first in the story
                    if delete_first_state:
                        raise InvalidStory(
                            f"Found two unpredictable actions in one story "
                            f"'{tracker.sender_id}'. Check your story files."
                        )
                    delete_first_state = True

                # reset entity_data for the the next turn
                entity_data = {}

            if delete_first_state:
                states = states[1:]

            trackers_as_states.append(states[:-1])
            trackers_as_actions.append(actions)
            trackers_as_entities.append(entities)

        self._remove_user_text_if_intent(trackers_as_states)

        return trackers_as_states, trackers_as_actions, trackers_as_entities

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain,
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

        Returns:
            A list of states.
        """
        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in trackers
        ]
        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

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

    def training_states_actions_and_entities(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, actions and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain

        Returns:
            A tuple of list of states, list of actions and list of entity data.
        """
        trackers_as_states = []
        trackers_as_actions = []
        trackers_as_entities = []

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
            entity_data = {}
            for event in tracker.applied_events():
                if isinstance(event, UserUttered):
                    entity_data = self._entity_data(event)

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
                        trackers_as_entities.append([entity_data])
                else:
                    trackers_as_states.append(sliced_states)
                    trackers_as_actions.append([event.action_name or event.action_text])
                    trackers_as_entities.append([entity_data])

                # reset entity_data for the the next turn
                entity_data = {}
                pbar.set_postfix({"# actions": "{:d}".format(len(trackers_as_actions))})

        self._remove_user_text_if_intent(trackers_as_states)

        logger.debug("Created {} action examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions, trackers_as_entities

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform
            domain: The domain
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.

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
        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

        return trackers_as_states
