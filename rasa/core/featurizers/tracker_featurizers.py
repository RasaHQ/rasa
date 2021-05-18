from pathlib import Path
from collections import defaultdict

import jsonpickle
import logging

from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Text, Union, Any, Iterator
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

    def __init__(self, message: Text) -> None:
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
    def _create_states(
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[State]:
        """Create states for the given tracker.

        Args:
            tracker: a :class:`rasa.core.trackers.DialogueStateTracker`
            domain: a :class:`rasa.shared.core.domain.Domain`
            omit_unset_slots: If `True` do not include the initial values of slots.
            ignore_rule_only_turns: If `True` ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            a list of states
        """
        return tracker.past_states(
            domain,
            omit_unset_slots=omit_unset_slots,
            ignore_rule_only_turns=ignore_rule_only_turns,
            rule_only_data=rule_only_data,
        )

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
        bilou_tagging: bool = False,
    ) -> List[List[Dict[Text, List["Features"]]]]:

        return [
            [
                self.state_featurizer.encode_entities(
                    entity_data, interpreter, bilou_tagging
                )
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
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, actions and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of actions and list of entity data.
        """
        rasa.shared.utils.io.raise_deprecation_warning(
            "'training_states_actions_and_entities' is being deprecated in favor of "
            "'training_states_labels_and_labels'."
        )
        raise NotImplementedError(
            f"`{self.__class__.__name__}` should implement how to encode trackers as feature vectors"
        )

    def training_states_and_actions(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and actions.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states and list of actions.
        """

        rasa.shared.utils.io.raise_deprecation_warning(
            "'training_states_and_actions' is being deprecated in favor of "
            "'training_states_and_labels'."
        )

        return self.training_states_and_labels(
            trackers, domain, omit_unset_slots=omit_unset_slots
        )

    def training_states_and_labels(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Transforms list of trackers to lists of states and labels.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states and list of labels.
        """
        (
            trackers_as_states,
            trackers_as_labels,
            _,
        ) = self.training_states_labels_and_entities(
            trackers, domain, omit_unset_slots=omit_unset_slots
        )
        return trackers_as_states, trackers_as_labels

    def training_states_labels_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, labels, and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of labels and list of entity data.
        """
        return self.training_states_actions_and_entities(
            trackers, domain, omit_unset_slots=omit_unset_slots
        )

    def featurize_trackers(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        bilou_tagging: bool = False,
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
            bilou_tagging: indicates whether BILOU tagging should be used or not

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

        self.state_featurizer.prepare_for_training(domain, interpreter, bilou_tagging)

        (
            trackers_as_states,
            trackers_as_labels,
            trackers_as_entities,
        ) = self.training_states_labels_and_entities(trackers, domain)

        tracker_state_features = self._featurize_states(trackers_as_states, interpreter)
        label_ids = self._convert_labels_to_ids(trackers_as_labels, domain)

        entity_tags = self._create_entity_tags(
            trackers_as_entities, interpreter, bilou_tagging
        )

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
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

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
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        """Create state features for prediction.

        Args:
            trackers: A list of state trackers
            domain: The domain
            interpreter: The interpreter
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list (corresponds to the list of trackers)
            of lists (corresponds to all dialogue turns)
            of dictionaries of state type (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
            ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
            turns in all trackers.
        """
        trackers_as_states = self.prediction_states(
            trackers,
            domain,
            use_text_for_last_user_input,
            ignore_rule_only_turns,
            rule_only_data,
        )
        return self._featurize_states(trackers_as_states, interpreter)

    def persist(self, path: Union[Text, Path]) -> None:
        """Persist the tracker featurizer to the given path.

        Args:
            path: The path to persist the tracker featurizer to.
        """
        featurizer_file = Path(path) / FEATURIZER_FILE
        rasa.shared.utils.io.create_directory_for_file(featurizer_file)

        # entity tags are persisted in TED policy, they are not needed for prediction
        if self.state_featurizer is not None:
            self.state_featurizer.entity_tag_specs = None

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
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, labels and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of labels and list of entity data.
        """
        rasa.shared.utils.io.raise_deprecation_warning(
            "'training_states_actions_and_entities' is being deprecated in "
            "favor of 'training_states_labels_and_entities'."
        )
        return self.training_states_labels_and_entities(
            trackers, domain, omit_unset_slots=omit_unset_slots
        )

    def training_states_labels_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of state, action labels and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

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
            states = self._create_states(
                tracker, domain, omit_unset_slots=omit_unset_slots
            )

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
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states.
        """
        trackers_as_states = [
            self._create_states(
                tracker,
                domain,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=rule_only_data,
            )
            for tracker in trackers
        ]
        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

        return trackers_as_states


class MaxHistoryTrackerFeaturizer(TrackerFeaturizer):
    """Slices the tracker history into max_history batches.

    Creates training data that uses last output for prediction.
    Training data is padded up to the max_history with -1.
    """

    LABEL_NAME = "action"

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
        tracker: DialogueStateTracker,
        states: List[State],
        labels: Optional[List[Text]] = None,
    ) -> int:
        """Hash states (and optionally label) for efficient deduplication.
        If labels is None, labels is not hashed.
        """

        frozen_states = tuple(
            s if s is None else tracker.freeze_current_state(s) for s in states
        )
        if labels is not None:
            frozen_labels = tuple(labels)
            return hash((frozen_states, frozen_labels))
        else:
            return hash(frozen_states)

    def training_states_actions_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, labels and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of labels and list of entity data.
        """
        rasa.shared.utils.io.raise_deprecation_warning(
            "'training_states_actions_and_entities' is being deprecated in "
            "favor of 'training_states_labels_and_entities'."
        )
        return self.training_states_labels_and_entities(
            trackers, domain, omit_unset_slots=omit_unset_slots
        )

    def training_states_labels_and_entities(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Tuple[List[List[State]], List[List[Text]], List[List[Dict[Text, Any]]]]:
        """Transforms list of trackers to lists of states, labels and entity data.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A tuple of list of states, list of labels and list of entity data.
        """

        self._setup_example_iterator()

        example_states = []
        example_labels = []
        example_entities = []

        logger.debug(
            "Creating states and {} label examples from "
            "collected trackers (by {}({}))..."
            "".format(
                self.LABEL_NAME,
                type(self).__name__,
                type(self.state_featurizer).__name__,
            )
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:

            for states, label, entities in self._example_iterator(
                tracker, domain, omit_unset_slots=omit_unset_slots
            ):

                if self._check_example_cache(tracker, states, label):
                    continue

                example_states.append(states)
                example_labels.append(label)
                example_entities.append(entities)

                pbar.set_postfix({f"# {self.LABEL_NAME}": f"{len(example_labels):d}"})

        self._cleanup_example_iterator()
        self._remove_user_text_if_intent(example_states)

        logger.debug(f"Created {len(example_states)} {self.LABEL_NAME} examples.")

        return example_states, example_labels, example_entities

    def _setup_example_iterator(self) -> None:
        """Create set for filtering out duplicated training examples."""
        if self.remove_duplicates:
            self.hashed_examples = set()

    def _example_iterator(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Iterator[Tuple[List[State], List[Text], List[Dict[Text, Any]]]]:
        """Create an iterator over the training examples that will be created
        from the provided tracker.

        Returns:
            An iterator over state, labels, entity tag tuples
        """

        tracker_states = self._create_states(
            tracker, domain, omit_unset_slots=omit_unset_slots
        )

        label_index = 0
        entity_data = {}
        for event in tracker.applied_events():
            if isinstance(event, UserUttered):
                entity_data = self._entity_data(event)

            elif isinstance(event, ActionExecuted):

                label_index += 1

                # use only actions which can be predicted at a stories start
                if event.unpredictable:
                    continue

                sliced_states = self.slice_state_history(
                    tracker_states[:label_index], self.max_history
                )
                label = [event.action_name or event.action_text]
                entities = [entity_data]

                yield sliced_states, label, entities

                # reset entity_data for the the next turn
                entity_data = {}

    def _check_example_cache(
        self, tracker: DialogueStateTracker, states: List[State], label: List[Text],
    ) -> bool:
        """Returns True if training example is a duplicate."""
        if not self.remove_duplicates:
            return False
        else:
            hashed = self._hash_example(tracker, states, label)
            if hashed not in self.hashed_examples:
                self.hashed_examples.add(hashed)
                return False
            else:
                return True

    def _cleanup_example_iterator(self) -> None:
        """Remove deduplication cache and remove intent text when intent label
        is used."""
        if self.remove_duplicates:
            self.hashed_examples = None

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states.
        """
        trackers_as_states = [
            self._create_states(
                tracker,
                domain,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=rule_only_data,
            )
            for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]
        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

        return trackers_as_states


class IntentMaxHistoryTrackerFeaturizer(MaxHistoryTrackerFeaturizer):

    LABEL_NAME = "intent"

    def _convert_labels_to_ids(
        self, trackers_as_intents: List[List[Text]], domain: Domain
    ) -> np.ndarray:
        """Convert a list of labels to an np.ndarray of label ids.
        The number of rows is equal to `len(trackers_as_intents)`.
        The number of columns is equal to the maximum number of labels
        that any Labels item has. Rows are padded with -1 if not all Labels
        items have the same number of labels.

        Returns:
            A 2d np.ndarray of label ids.
        """

        # store labels in numpy arrays so that it corresponds to np arrays
        # of input features
        label_ids = [
            [domain.intents.index(intent) for intent in tracker_intents]
            for tracker_intents in trackers_as_intents
        ]

        pad_val = -1

        # Add -1 padding to labels array so that
        # each example has equal number of labels
        multiple_labels_count = [len(a) for a in label_ids]
        max_labels_count = max(multiple_labels_count)
        num_padding_needed = [max_labels_count - len(a) for a in label_ids]

        new_label_ids = []
        for ids, num_pads in zip(label_ids, num_padding_needed):
            if num_pads:
                ids.extend([pad_val] * num_pads)
            new_label_ids.append(ids)

        new_label_ids = np.array(new_label_ids)
        return new_label_ids

    def _setup_example_iterator(self) -> None:
        """Create any data structures for deduplication and tracking multiple
        intent labels.
        """
        super()._setup_example_iterator()
        self._state_hash_to_labels = defaultdict(list)

    def _example_iterator(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        omit_unset_slots: bool = False,
    ) -> Iterator[Tuple[List[State], List[Text], List[Dict[Text, Any]]]]:
        """Create an iterator over the training examples that will be created
        from the provided tracker.

        Returns:
            An iterator over state, labels, entity tag tuples
        """

        tracker_states = self._create_states(
            tracker, domain, omit_unset_slots=omit_unset_slots
        )

        label_index = 0
        for event in tracker.applied_events():

            if isinstance(event, ActionExecuted):
                label_index += 1

            elif isinstance(event, UserUttered):

                sliced_states = self.slice_state_history(
                    tracker_states[:label_index], self.max_history
                )
                label = [event.intent_name or event.intent_text]
                entities = [{}]

                yield sliced_states, label, entities

    def _check_example_cache(
        self, tracker: DialogueStateTracker, states: List[State], label: List[Text],
    ) -> bool:
        if not super()._check_example_cache(tracker, states, label):
            state_hash = self._hash_example(tracker, states)
            self._state_hash_to_labels[state_hash].append(label)
            return False
        else:
            return True

    def _cleanup_example_iterator(self) -> None:
        """Clean up cache data structures and finalize any training labels.

        Collects all positive intent labels for a given state hash and adds
        them to the original label for each state hash in the training data.
        """

        for labelset in self._state_hash_to_labels.values():
            # Get the set of labels associated with the state hash.
            codomain = set([labels[0] for labels in labelset])
            for labels in labelset:
                # Remove the duplicate label in the first position
                # and update the positive labels.
                filtered_codomain = filter(lambda label: label != labels[0], codomain)
                labels.extend(filtered_codomain)

        self._state_hash_to_labels = None
        super()._cleanup_example_iterator()

    @staticmethod
    def _cleanup_last_user_state_with_action_listen(trackers_as_states):
        """Clean up the last user state where previous action is `action_listen`.

        Args:
            trackers_as_states: Trackers converted to states

        Returns:
            Filtered states with last `action_listen` removed.
        """
        for states in trackers_as_states:
            last_state = states[-1]
            if is_prev_action_listen_in_state(last_state):
                del states[-1]

        return trackers_as_states

    def prediction_states(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of states for prediction.

        Args:
            trackers: The trackers to transform.
            domain: The domain.
            use_text_for_last_user_input: Indicates whether to use text or intent label
                for featurizing last user input.
            ignore_rule_only_turns: If True ignore dialogue turns that are present
                only in rules.
            rule_only_data: Slots and loops,
                which only occur in rules but not in stories.

        Returns:
            A list of states.
        """
        trackers_as_states = [
            self._create_states(
                tracker,
                domain,
                ignore_rule_only_turns=ignore_rule_only_turns,
                rule_only_data=rule_only_data,
            )
            for tracker in trackers
        ]
        trackers_as_states = [
            self.slice_state_history(states, self.max_history)
            for states in trackers_as_states
        ]
        self._choose_last_user_input(trackers_as_states, use_text_for_last_user_input)

        # `tracker_as_states` contain a state with intent = last intent
        # and previous action = action_listen. This state needs to be
        # removed as it was not present during training as well because
        # predicting the last intent is what the policies using this
        # featurizer do.
        self._cleanup_last_user_state_with_action_listen(trackers_as_states)

        return trackers_as_states
