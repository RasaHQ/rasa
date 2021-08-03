import jsonpickle
import logging
import copy
from numpy.lib.arraysetops import isin

from tqdm import tqdm
from typing import Callable, Tuple, List, Optional, Dict, Text, Any, Set
import numpy as np

from rasa.architecture_prototype.interfaces import ComponentPersistorInterface
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.core.domain import State, Domain, SubState
from rasa.shared.core.events import ActionExecuted, Event, UserUttered
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.constants import PREVIOUS_ACTION, USER
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES, ACTION_NAME, ACTION_TEXT
from rasa.shared.exceptions import RasaException
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)

EXTRACTOR_FILE = "extractor.json"
FEATURIZER_FILE = "featurizer.json"


def copy_state(
    state: State,
    sub_state_type_attribute_combinations: Optional[Dict[Text, List[Text]]] = None,
) -> State:
    """Creates a copy that only contains certain substate types and attributes.
    """
    if sub_state_type_attribute_combinations is None:
        return copy.deepcopy(state)
    partial_state = dict()
    for sub_state_type, attributes in sub_state_type_attribute_combinations.items():
        sub_state = dict()
        if sub_state_type in state and attributes:
            attributes_left = set(sub_state.keys()).intersection(attributes)
            for attribute in attributes_left:
                sub_state[attribute] = copy.deepcpoy(state[sub_state_type][attribute])
            partial_state[sub_state] = sub_state
    return partial_state


def get_events_and_states(
    tracker: DialogueStateTracker, domain: Domain, omit_unset_slots: bool
):
    """Extracts and aligns events and states from the given tracker.

    The logic behind the computed alignment is based on the relation between
    a trackers `past_states` and its `applied_events`:
    (0)
    (1) Every encounter of an `ActionExecuted` in the `applied_events`
        triggers the creation of a
    state, which captures the events up to *right before the event that
    triggered its creation*.
    Additionally, the creation of the last state event is triggered after
    all events have been passed. Hence this last state includes information
    of the very last applied_event.

    Note that this function retrieves `past_states` from the tracker *without*
    ignoring rule only turns!

    References:
        - tracker.generate_all_prior_trackers
        - domain.states_for_tracker_history begin called *without skipping
        rule turns* !

    Returns:
        a list of all applied events from the given tracker,
        a list of all `past_states` of the tracker (without skipping any turns),
        a list of tuples containing an state and a event where the event is the
        last piece of information captured in the corresponding state
    """
    states = tracker.past_states(
        domain,
        omit_unset_slots=omit_unset_slots,
        ignore_rule_only_turns=False,
        rule_only_data=None,
    )
    applied_events = tracker.applied_events()

    # Sanity Check: There are no "BotUttered" events or so - every "applied_event"
    # is an ActionExecuted event
    assert all(
        isinstance(event, UserUttered) or isinstance(event, ActionExecuted)
        for event in applied_events
    )

    # Sanity Check: We should have 1 more state than ActionExecuted events.
    num_action_executed_events = sum(
        1 for event in applied_events if isinstance(event, ActionExecuted)
    )
    assert num_action_executed_events + 1 == len(states)

    # Sanity Check: After every UserUttered event there is an ActionExecuted event
    for idx, event in enumerate(applied_events):
        if idx < len(applied_events) - 1:
            if isinstance(event, UserUttered):
                assert isinstance(applied_events[idx + 1], ActionExecuted)

    # Align States and the last Event that they *do* describe.
    alignment: List[Tuple[Event, State]] = []
    skipped_one_state = False
    state_idx = -1
    for trigger_event_idx, trigger_event in enumerate(applied_events + [None]):
        # where we use `None`s as dummies to capture the last state as well
        if isinstance(trigger_event, ActionExecuted) or trigger_event is None:
            state_idx = state_idx + 1  # i.e. idx of state created by trigger_event
            if trigger_event_idx < 1:
                # State would describe nothing because nothing has happend before
                # so we just pass...
                skipped_one_state = True
                continue
            else:
                # state[idx] includes knowledge about the previous event, so...
                pair = (states[state_idx], applied_events[trigger_event_idx - 1])
                alignment.append(pair)

    # Sanity Check: We included the last state and event
    assert alignment[-1][0] == states[-1]
    assert alignment[-1][1] == applied_events[-1]

    # Sanity Check: There should be one (state, event) tuple per state - except
    # maybe the first state (which would be empty)
    assert len(alignment) == len(states) - skipped_one_state

    return applied_events, states, alignment


class InvalidStory(RasaException):
    """Exception that can be raised if story cannot be featurized."""

    def __init__(self, message: Text) -> None:
        self.message = message
        super(InvalidStory, self).__init__()

    def __str__(self) -> Text:
        return self.message


class TrackerStateExtractor:
    """
    """

    def __init__(
        self,
        is_target: Optional[Callable[[Event], bool]] = None,
        hide_step: Optional[Callable[[Event, State], bool]] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
        persistor: Optional[ComponentPersistorInterface] = None,
    ) -> None:
        """Initialize the tracker featurizer.

        Args:
            custom_is_target:
            hide_step:
            state_featurizer: The state featurizer used to encode the states.
            max_history: if set to None, the complete history is taken into account;
              otherwise only the last `max_history`+1 states will be extracted (note
              that the +1 represents the target, hence your policy will use the last
              `max_history` many states as input)
            remove_duplicates: set to `True` if you want to ignore event sequences
               which are identical (i.e. no state sequences will be created from these
               event sequences)
        """
        self._persistor = persistor
        self.max_history = max_history
        self.remove_duplicates = remove_duplicates
        self.custom_is_target = is_target
        self.hide_step = hide_step

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"is_target={repr(self.custom_is_target)}, "
            f"hide_step={repr(self.hide_step)}, "
            f"max_history={self.max_history}, "
            f"remove_duplicates={self.remove_duplicates}, "
            f"persistor={repr(self._persistor)})"
        )

    @staticmethod
    def slice_state_history(
        states: List[State], slice_length: Optional[int]
    ) -> List[State]:
        """At most the last slice_length states from the given list.

        Args:
            states: The states
            slice_length: the slice length or None

        Returns:
            sub-sequence of the given list of states if slice_length is defined;
            otherwise just the given list of states
        """
        if not slice_length:
            return states

        return states[-slice_length:]

    @staticmethod
    def _hash_example(states: List[State]) -> int:
        """Hash states for efficient deduplication."""
        # FIXME: these hashes change with a new session - is that a problem?
        frozen_states = tuple(
            s if s is None else DialogueStateTracker.freeze_current_state(s)
            for s in states
        )
        return hash((frozen_states,))

    @staticmethod
    def _remove_user_text_if_intent(states: List[State]) -> None:
        for state in states:
            # remove text features to only use intent
            if state.get(USER, {}).get(INTENT) and state.get(USER, {}).get(TEXT):
                del state[USER][TEXT]

    def is_target(self, event: Event) -> bool:
        return (
            isinstance(event, ActionExecuted)
            if self.custom_is_target is None
            else self.custom_is_target(event)
        )

    def _extract_states_from_tracker_for_training(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        max_training_examples: Optional[int] = None,
        omit_unset_slots: bool = False,
        deduplication_pool: Optional[Set] = None,
    ) -> List[List[State]]:
        """Transform a tracker to a list of state sequences.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.
            deduplication_pool: If some set is given, this will be used to store
              hashes of state sequences to be able to avoid repeating any state
              sequence. Hashes of new state sequences create by this method will
              be added to the `deduplication_pool`.

        Returns:
            A list of state sequences.
        """
        result: List[List[State]] = []
        if max_training_examples is not None and max_training_examples <= 0:
            return result

        _, _, states_with_lastest_event = get_events_and_states(
            tracker=tracker, domain=domain, omit_unset_slots=omit_unset_slots
        )

        if self.hide_step:
            states_with_lastest_event = [
                (state, event)
                for (state, event) in states_with_lastest_event
                if not self.hide_step(event, state)
            ]

        states, events = list(zip(*states_with_lastest_event))

        for idx, event in enumerate(events):

            if idx > 0 and self.is_target(
                event
            ):  # TODO: is idx>0 ok as a replacement for not event.unpredictable (?)

                sliced_states = self.slice_state_history(
                    states[: (idx + 1)],  # i.e. include state belonging to event
                    self.max_history
                    if (self.max_history is None)
                    else (self.max_history + 1),
                )
                if self.remove_duplicates:
                    # only continue with tracker_states that created a
                    # hashed_featurization we haven't observed
                    hashed = self._hash_example(sliced_states)
                    if hashed not in deduplication_pool:
                        deduplication_pool.add(hashed)
                        result.append(sliced_states)
                else:
                    result.append(sliced_states)

                if (
                    max_training_examples is not None
                    and len(result) >= max_training_examples
                ):
                    break
        return result

    def extract_states_from_trackers_for_training(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        max_training_examples: Optional[int] = None,
        omit_unset_slots: bool = False,
    ) -> List[List[State]]:
        """Transforms list of trackers to lists of state sequences.

        Args:
            trackers: The trackers to transform
            domain: The domain
            omit_unset_slots: If `True` do not include the initial values of slots.

        Returns:
            A list of state sequences.
        """
        list_of_state_sequences: List[List[State]] = []

        # from multiple states that create equal featurizations
        # we only keep one - if de-duplication is enabled
        hashed_examples = set() if self.remove_duplicates else None

        logger.debug(
            f"Creating states and action examples from "
            f"collected trackers (by {self})..."
        )
        pbar = tqdm(
            trackers,
            desc="Processed trackers",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for tracker in pbar:
            list_of_state_sequences_from_tracker = self._extract_states_from_tracker_for_training(
                tracker=tracker,
                domain=domain,
                max_training_examples=max_training_examples,
                omit_unset_slots=omit_unset_slots,
                deduplication_pool=hashed_examples,
            )
            list_of_state_sequences.extend(list_of_state_sequences_from_tracker)

            if max_training_examples is not None:
                max_training_examples = max_training_examples - len(
                    list_of_state_sequences
                )
                if max_training_examples <= 0:
                    break

        # NOTE: the remove text if intent part needs to be moved to encode_state

        return list_of_state_sequences

    def extract_states_and_actions_for_training(
        self,
        trackers: List[DialogueStateTracker],
        domain: Domain,
        omit_unset_slots: bool,
        max_training_examples: Optional[int] = None,
    ) -> Tuple[List[List[State]], List[List[Text]]]:
        """Creates unfeaturized training data for e.g. rule/memoization policy.

        """
        list_of_state_sequences = self.extract_states_from_trackers_for_training(
            trackers,
            domain,
            omit_unset_slots=omit_unset_slots,
            max_training_examples=max_training_examples,
        )
        inputs = []
        outputs = []
        for state_sequence in list_of_state_sequences:
            inputs.append(state_sequence[:-1])
            last_action = state_sequence[-1][PREVIOUS_ACTION]
            last_action_name_or_text = (
                last_action.get(ACTION_NAME, None) or last_action[ACTION_TEXT]
            )
            outputs.append([last_action_name_or_text])
        return inputs, outputs

    def extract_states_from_tracker_for_prediction(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[State]]:
        """Transforms a tracker to a lists of states for prediction.

        Args:
            tracker: The tracker to transform.
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
        states = tracker.past_states(
            domain,
            omit_unset_slots=False,
            ignore_rule_only_turns=ignore_rule_only_turns,
            rule_only_data=rule_only_data,
        )
        states = self.slice_state_history(states, self.max_history)
        # FIXME/TODO: why does this happen after slicing?
        self._cleanup_states_for_prediction(states, use_text_for_last_user_input)
        return states

    def _cleanup_states_for_prediction(
        self, states: List[State], use_text_for_last_user_input: bool
    ) -> None:
        """Prepares a sequence of states that was extracted for prediction.

        """
        last_state = states[-1]

        # (1) only update the state of the real user utterance
        if not is_prev_action_listen_in_state(last_state):
            return

        # (2) keep either intent+entities or text only for the *last state*
        #     depending on the given flag
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

        # (3) for all other states, *always* remove the text if an intent is present

        # FIXME: if those messages passed through NLU pipeline before the policy
        # prediction pipeline, then we'll always have intents here? (unless it's for
        # an e2e policy)
        self._remove_user_text_if_intent(states)

    def persist(self) -> None:
        """Persist the tracker featurizer to the given path.

        Args:
            path: The path to persist the tracker featurizer to.
        """
        featurizer_file = self._persistor.file_for(FEATURIZER_FILE)
        # noinspection PyTypeChecker
        rasa.shared.utils.io.write_text_file(
            str(jsonpickle.encode(self)), featurizer_file
        )

    @staticmethod
    def load(
        persistor: ComponentPersistorInterface, resource_name: Text
    ) -> Optional["TrackerFeaturizer"]:
        """Load the featurizer from file.

        Args:
            path: The path to load the tracker featurizer from.

        Returns:
            The loaded tracker featurizer.
        """
        featurizer_file = persistor.get_resource(resource_name, FEATURIZER_FILE)
        return jsonpickle.decode(rasa.shared.utils.io.read_file(featurizer_file))


class TrackerFeaturizer:
    """
    """

    def __init__(
        self,
        is_target: Optional[Callable[[Event], bool]] = None,
        hide_step: Optional[Callable[[Event, State], bool]] = None,
        max_history: Optional[int] = None,
        remove_duplicates: bool = True,
        persistor: Optional[ComponentPersistorInterface] = None,
    ) -> None:
        """Initialize the tracker featurizer.

        Args:
            is_target: ...
            hide_step: ...
            max_history: if set to None, the complete history is taken into account;
              otherwise only the last `max_history`+1 states will be extracted (note
              that the +1 represents the target, hence your policy will use the last
              `max_history` many states as input)
            remove_duplicates: set to `True` if you want to ignore event sequences
               which are identical (i.e. no state sequences will be created from these
               event sequences)
        """
        self.tracker_state_extractor = TrackerStateExtractor(
            is_target=is_target,
            hide_step=hide_step,
            max_history=max_history,
            remove_duplicates=remove_duplicates,
            persistor=persistor,
        )
        self._persistor = persistor

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(state_extractor={repr(self.tracker_state_extractor)}, "
            f"state_featurizer={repr(self.state_featurizer)})"
        )

    def is_setup(self) -> bool:
        return self._is_setup

    def setup(self, domain: Domain, bilou_tagging: bool,) -> None:
        """This lazy setup is needed because the policies only know e.g. the domain
        once a policies `train()` is called but that happens after we instantiated
        the tracker featurizer (which we could change as well...).
        """
        self.state_featurizer = SingleStateFeaturizer(
            domain=domain, bilou_tagging=bilou_tagging
        )
        self.domain = domain
        self._is_setup = True

    def raise_error_if_not_setup(self):
        if not self._is_setup:
            raise RuntimeError(
                f"Expected this {self.__class__.__name__} instance to be setup for "
                f"a specific domain and bilou_tagging mode. You can achieve this by "
                f"calling `setup()` before attempting any featurization."
            )

    def featurize_trackers_for_training(
        self,
        trackers: List[DialogueStateTracker],
        e2e_features: Optional[Dict[Text, Message]] = None,
        omit_unset_slots: bool = False,
        targets_include_actions: bool = True,
        targets_include_entities: bool = True,
        targets_include_intents: bool = True,
        max_training_examples: Optional[int] = None,
    ) -> Tuple[
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
        List[List[Dict[Text, List[Features]]]],
        np.ndarray,
    ]:
        """Creates training data that contains `Features` from the NLU pipeline.

        Args:
            trackers: list of training trackers
            domain: the domain
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
        self.raise_error_if_not_setup()

        list_of_state_sequences: List[
            List[State]
        ] = self.tracker_state_extractor.extract_states_from_trackers_for_training(
            trackers,
            domain=self.domain,
            omit_unset_slots=omit_unset_slots,
            max_training_examples=max_training_examples,
        )

        inputs = []
        entities = []
        actions = []
        intents = []
        for state_sequence in list_of_state_sequences:
            inputs.append([])
            entities.append([])
            actions_for_state_sequence = np.zeros(len(state_sequence) - 1)
            intents_for_state_sequence = np.zeros(len(state_sequence) - 1)
            for idx, state in enumerate(state_sequence):
                state_encoding = self.state_featurizer.encode_state(
                    state,
                    e2e_features,
                    targets_include_actions=targets_include_actions,
                    targets_include_entities=targets_include_entities,
                    targets_include_intents=targets_include_intents,
                )
                if idx < len(state_sequence) - 1:
                    inputs[-1].append(state_encoding["input"])
                if idx > 0:
                    if targets_include_entities:
                        entities[-1].append(state_encoding["target_entity"])
                    for (flag, array, key) in [
                        (
                            targets_include_actions,
                            actions_for_state_sequence,
                            "target_action",
                        ),
                        (
                            targets_include_intents,
                            intents_for_state_sequence,
                            "target_intent",
                        ),
                    ]:
                        if flag:
                            array[idx - 1] = state_encoding[key]
            for (flag, array, collection) in [
                (targets_include_actions, actions_for_state_sequence, actions),
                (targets_include_intents, intents_for_state_sequence, intents),
            ]:
                if flag:
                    collection.append(np.array(array))
        actions = np.array(actions)

        return (inputs, actions, entities, intents)

    def featurize_trackers_for_prediction(
        self,
        trackers: List[DialogueStateTracker],
        e2e_features: Dict[Text, Message],
        use_text_for_last_user_input: bool = False,
        ignore_rule_only_turns: bool = False,
        rule_only_data: Optional[Dict[Text, Any]] = None,
    ) -> List[List[Dict[Text, List[Features]]]]:
        """Create state features for prediction.

        Args:
            trackers: A list of state trackers
            domain: The domain
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
        output = []
        for tracker in trackers:
            states = self.tracker_state_extractor.extract_states_from_tracker_for_prediction(
                tracker,
                self.domain,
                use_text_for_last_user_input,
                ignore_rule_only_turns,
                rule_only_data,
            )
            list_of_state_encodings = [
                self.state_featurizer.encode_state(
                    state=state,
                    e2e_features=e2e_features,
                    targets_include_actions=False,
                    targets_include_entities=False,
                )
                for state in states
            ]
            output.append(
                [state_encoding["input"] for state_encoding in list_of_state_encodings]
            )
        return output

    def persist(self) -> None:
        """Persist the tracker featurizer to the given path.

        Args:
            path: The path to persist the tracker featurizer to.
        """
        featurizer_file = self._persistor.file_for(FEATURIZER_FILE)

        # # entity tags are persisted in TED policy, they are not needed for prediction # FIXME: ?
        # if self.state_featurizer is not None:
        #     self.state_featurizer.entity_tag_specs = None

        # noinspection PyTypeChecker
        rasa.shared.utils.io.write_text_file(
            str(jsonpickle.encode(self)), featurizer_file
        )

    @staticmethod
    def load(
        persistor: ComponentPersistorInterface, resource_name: Text
    ) -> Optional["TrackerFeaturizer"]:
        """Load the featurizer from file.

        Args:
            path: The path to load the tracker featurizer from.

        Returns:
            The loaded tracker featurizer.
        """
        featurizer_file = persistor.get_resource(resource_name, FEATURIZER_FILE)
        return jsonpickle.decode(rasa.shared.utils.io.read_file(featurizer_file))


# NOTE: FullDialogueFeaturizer wasn't used anymore and *Intent* version can be
# replaced by adding options to the above
