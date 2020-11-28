from pathlib import Path

import jsonpickle
import logging

from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Union, Any
import numpy as np

from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import State, Domain
from rasa.shared.core.events import ActionExecuted, UserUttered, UserUtteranceReverted
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

        # TODO: Revert changes here back to original
        featurized = []
        for tracker_states in trackers_as_states:
            state_features = []
            for state in tracker_states:
                state_features.append(
                    self.state_featurizer.encode_state(state, interpreter)
                )
            featurized.append(state_features)
        return featurized

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
        if event.text:
            return {TEXT: event.text, ENTITIES: event.entities}

        # input is not textual, so add empty dict
        return {}

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
            "Featurizer must have the capacity to encode trackers to feature vectors"
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

    @staticmethod
    def _choose_last_user_input(
        trackers_as_states: List[List[State]], use_text_for_last_user_input: bool
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
            else:
                # remove text features to only use intent
                if last_state.get(USER, {}).get(TEXT):
                    del last_state[USER][TEXT]

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
            use_text_for_last_user_input: boolean

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
            use_text_for_last_user_input: boolean

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
            use_text_for_last_user_input: boolean

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
            use_text_for_last_user_input: boolean

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


class IntentMaxHistoryFeaturizer(MaxHistoryTrackerFeaturizer):
    @staticmethod
    def _hash_states(states: List[State], tracker: DialogueStateTracker) -> int:
        """Hash states for efficient collection of multiple labels."""
        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        return hash(frozen_states)

    def training_states_and_actions(
        self, trackers: List[DialogueStateTracker], domain: Domain
    ) -> Tuple[List[List[State]], List[List[Text]], List[bool]]:
        """Transforms list of trackers to lists of states and actions.
        Training data is padded up to the max_history with -1.
        """

        trackers_as_states = []
        trackers_as_actions = []
        is_label_from_rules = []

        # from multiple states that create equal featurizations
        # we only need to keep one.
        hashed_examples = set()

        # We keep all the unique sequence of states hashed
        # so that we can efficiently collect multiple labels for each unique sequence.
        hashed_states = set()

        states_indices = dict()

        logger.debug(
            "Creating states and intent examples from "
            "collected trackers (by {}({}))..."
            "".format(type(self).__name__, type(self.state_featurizer).__name__)
        )
        # print(len(trackers))

        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )

        example_index = 0

        for tracker in pbar:

            is_rule_tracker = tracker.is_rule_tracker

            states = self._create_states(tracker, domain)

            states_length_for_intent = 0
            for event in tracker.applied_events():
                if isinstance(event, ActionExecuted):
                    states_length_for_intent += 1

                if isinstance(event, UserUttered):

                    sliced_states = self.slice_state_history(
                        states[:states_length_for_intent], self.max_history
                    )

                    if self.remove_duplicates:
                        hashed_example = self._hash_example(
                            sliced_states, event.intent_name, tracker
                        )

                        hashed_state = self._hash_states(sliced_states, tracker)
                        # print("hash of state", hashed_state)

                        # only continue with tracker_states that created a
                        # hashed_featurization we haven't observed
                        if hashed_example not in hashed_examples:
                            hashed_examples.add(hashed_example)

                            # First create a new data point for this pair
                            trackers_as_states.append(sliced_states)
                            trackers_as_actions.append([event.intent_name])
                            is_label_from_rules.append(is_rule_tracker)

                            # print(list(zip(trackers_as_states, trackers_as_actions)))

                            self._update_labels_for_all_states(
                                event.intent_name,
                                example_index,
                                hashed_state,
                                hashed_states,
                                states_indices,
                                trackers_as_actions,
                            )
                            # print(list(zip(trackers_as_states, trackers_as_actions)))
                            # print(states_indices)
                            example_index += 1
                    else:
                        hashed_state = self._hash_states(sliced_states, tracker)
                        # First create a new data point for this pair
                        trackers_as_states.append(sliced_states)
                        trackers_as_actions.append([event.intent_name])
                        is_label_from_rules.append(is_rule_tracker)

                        self._update_labels_for_all_states(
                            event.intent_name,
                            example_index,
                            hashed_state,
                            hashed_states,
                            states_indices,
                            trackers_as_actions,
                        )
                        example_index += 1
                    # print("======================")

                    pbar.set_postfix(
                        {"# intents": "{:d}".format(len(trackers_as_actions))}
                    )

        logger.debug("Created {} intent examples.".format(len(trackers_as_actions)))

        return trackers_as_states, trackers_as_actions, is_label_from_rules

    @staticmethod
    def _update_labels_for_all_states(
        label,
        example_index,
        hashed_state,
        hashed_states,
        states_indices,
        trackers_as_actions,
    ):
        if hashed_state not in hashed_states:
            # The sequence of states is an unseen one
            hashed_states.add(hashed_state)
            states_indices[hashed_state] = [example_index]

            # trackers_as_states.append(sliced_states)
            # trackers_as_actions.append([event.intent["name"]])

        else:
            # We have seen this sequence of state before, so append the new intent
            # label in all previously seen states and for the new example created
            # add the already seen labels for this sequence.

            # First get the new example updated. It must have been appended at the end.
            trackers_as_actions[-1].extend(
                trackers_as_actions[states_indices[hashed_state][0]]
            )

            # print("updated new example")
            # print(trackers_as_actions)

            # print("updating other indices", states_indices[hashed_state])
            # Get all the other examples with same sequence of states updated
            for index in states_indices[hashed_state]:
                # print(index, trackers_as_actions[index])
                trackers_as_actions[index].append(label)

            # Update the indices where this state occurs
            states_indices[hashed_state].append(example_index)

        # print("updates states indices", states_indices)

    @staticmethod
    def _get_label_index_after_padding(original_index):
        return original_index + 1

    def _convert_labels_to_ids(
        self, trackers_as_intents: List[List[Text]], domain: Domain
    ) -> np.ndarray:

        # store labels in numpy arrays so that it corresponds to np arrays of input features
        label_ids = [
            [
                self._get_label_index_after_padding(domain.index_for_intent(intent))
                for intent in tracker_intents
            ]
            for tracker_intents in trackers_as_intents
        ]

        # new_label_ids = label_ids
        multiple_labels_count = [len(a) for a in label_ids]
        max_labels_count = max(multiple_labels_count)
        paddings_needed = [max_labels_count - len(a) for a in label_ids]
        # print(paddings_needed)
        new_label_ids = []
        for ids, num_pads in zip(label_ids, paddings_needed):
            # print(ids, num_pads)
            if num_pads:
                ids.extend([0] * num_pads)
            new_label_ids.append(ids)
        #
        new_label_ids = np.array(new_label_ids)
        return new_label_ids

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
            use_text_for_last_user_input: boolean

        Returns:
            A list of states.
        """
        # Create a copy of trackers
        duplicate_trackers = [tracker.copy() for tracker in trackers]

        # Remove last user event
        for tracker in duplicate_trackers:
            tracker.update(UserUtteranceReverted(), domain)

        trackers_as_states = [
            self._create_states(tracker, domain) for tracker in duplicate_trackers
        ]

        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]

        # print(trackers_as_states)

        return trackers_as_states

    @staticmethod
    def _serialize_state_feature(features_list):

        feats = []
        for state in features_list:
            new_state = {}
            for key, val in state.items():
                new_state[key] = []
                for feat in val:
                    new_state[key].append(feat.features.toarray().tolist())
            feats.append(new_state)
        return feats

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> Tuple[
        List[List[Dict[Text, List["Features"]]]],
        np.ndarray,
        List[List[Dict[Text, List["Features"]]]],
        List[bool],
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

        self.state_featurizer.prepare_for_training(domain, interpreter)

        (
            trackers_as_states,
            trackers_as_actions,
            is_label_from_rules,
        ) = self.training_states_and_actions(trackers, domain)

        # print("Featurizing states")

        tracker_state_features = self._featurize_states(trackers_as_states, interpreter)

        # print("Featurizing labels")
        label_ids = self._convert_labels_to_ids(trackers_as_actions, domain)

        # print(label_ids)
        # print(label_ids.shape)
        #
        # multiple_labels_count = [a.shape[0] for a in label_ids]
        # max_labels_count = max(multiple_labels_count)
        # paddings_needed = [(0, max_labels_count - a.shape[0]) for a in label_ids]
        # label_ids = np.pad(label_ids, paddings_needed, mode="constant", constant_values=-1, axis=1)
        #
        # print(label_ids)
        # exit(0)

        # for tracker_states, tracker_actions, label_id in zip(trackers_as_states, trackers_as_actions, label_ids):
        #     print(tracker_states, tracker_actions, label_id)

        # exit(0)

        # label_ids = self._collect_multiple_labels(tracker_state_features, label_ids)

        # print(domain.intents)
        # print(domain.action_names)
        # print(label_ids)

        return tracker_state_features, label_ids, [], is_label_from_rules
