from collections import defaultdict
import logging
import json
from typing import DefaultDict, Dict, Generator, List, NamedTuple, Optional, Text, Tuple

from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    PREVIOUS_ACTION,
    ACTION_UNLIKELY_INTENT_NAME,
    USER,
)
from rasa.shared.core.domain import Domain, State
from rasa.shared.core.events import ActionExecuted, Event
from rasa.shared.core.generator import TrackerWithCachedStates

from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class StoryConflict:
    """Represents a conflict between two or more stories.

    Here, a conflict means that different actions are supposed to follow from
    the same dialogue state, which most policies cannot learn.
    """

    def __init__(self, sliced_states: List[State]) -> None:
        """
        Creates a `StoryConflict` from a given state.

        Args:
            sliced_states: The (sliced) dialogue state at which the conflict occurs.
        """

        self._sliced_states = sliced_states
        # A list of actions that all follow from the same state.
        self._conflicting_actions: DefaultDict[Text, List[Text]] = defaultdict(
            list
        )  # {"action": ["story_1", ...], ...}

    def __hash__(self) -> int:
        return hash(str(list(self._sliced_states)))

    def add_conflicting_action(self, action: Text, story_name: Text) -> None:
        """Adds another action that follows from the same state.

        Args:
            action: Name of the action.
            story_name: Name of the story where this action is chosen.
        """
        self._conflicting_actions[action] += [story_name]

    @property
    def conflicting_actions(self) -> List[Text]:
        """List of conflicting actions.

        Returns:
            List of conflicting actions.

        """
        return list(self._conflicting_actions.keys())

    @property
    def conflict_has_prior_events(self) -> bool:
        """Checks if prior events exist.

        Returns:
            `True` if anything has happened before this conflict, otherwise `False`.
        """
        return _get_previous_event(self._sliced_states[-1])[0] is not None

    def __str__(self) -> Text:
        # Describe where the conflict occurs in the stories
        last_event_type, last_event_name = _get_previous_event(self._sliced_states[-1])
        if last_event_type:
            conflict_message = (
                f"Story structure conflict after {last_event_type} "
                f"'{last_event_name}':\n"
            )
        else:
            conflict_message = "Story structure conflict at the beginning of stories:\n"

        # List which stories are in conflict with one another
        for action, stories in self._conflicting_actions.items():
            conflict_message += (
                f"  {self._summarize_conflicting_actions(action, stories)}"
            )

        return conflict_message

    @staticmethod
    def _summarize_conflicting_actions(action: Text, stories: List[Text]) -> Text:
        """Gives a summarized textual description of where one action occurs.

        Args:
            action: The name of the action.
            stories: The stories in which the action occurs.

        Returns:
            A textural summary.
        """
        if len(stories) > 3:
            # Four or more stories are present
            conflict_description = (
                f"'{stories[0]}', '{stories[1]}', and {len(stories) - 2} other trackers"
            )
        elif len(stories) == 3:
            conflict_description = f"'{stories[0]}', '{stories[1]}', and '{stories[2]}'"
        elif len(stories) == 2:
            conflict_description = f"'{stories[0]}' and '{stories[1]}'"
        elif len(stories) == 1:
            conflict_description = f"'{stories[0]}'"
        else:
            raise ValueError(
                "An internal error occurred while trying to summarise a conflict "
                "without stories. Please file a bug report at "
                "https://github.com/RasaHQ/rasa."
            )

        return f"{action} predicted in {conflict_description}\n"


class TrackerEventStateTuple(NamedTuple):
    """Holds a tracker, an event, and sliced states associated with those."""

    tracker: TrackerWithCachedStates
    event: Event
    sliced_states: List[State]

    @property
    def sliced_states_hash(self) -> int:
        """Returns the hash of the sliced states."""
        return hash(json.dumps(self.sliced_states, sort_keys=True))


def find_story_conflicts(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int] = None,
) -> List[StoryConflict]:
    """Generates `StoryConflict` objects, describing conflicts in the given trackers.

    Args:
        trackers: Trackers in which to search for conflicts.
        domain: The domain.
        max_history: The maximum history length to be taken into account.

    Returns:
        StoryConflict objects.
    """
    if max_history:
        logger.info(
            f"Considering the preceding {max_history} turns for conflict analysis."
        )
    else:
        logger.info("Considering all preceding turns for conflict analysis.")

    # We do this in two steps, to reduce memory consumption:

    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    conflicting_state_action_mapping = _find_conflicting_states(
        trackers, domain, max_history
    )

    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = _build_conflicts_from_states(
        trackers, domain, max_history, conflicting_state_action_mapping
    )

    return conflicts


def _find_conflicting_states(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int],
    tokenizer: Optional[Tokenizer] = None,
) -> Dict[int, List[int]]:
    """Identifies all states from which different actions follow.

    Args:
        trackers: Trackers that contain the states.
        domain: The domain object.
        max_history: Number of turns to take into account for the state descriptions.
        tokenizer: A tokenizer to tokenize the user messages.

    Returns:
        A dictionary mapping state-hashes to a list of actions that follow from each
        state.
    """
    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    state_action_mapping: DefaultDict[int, List[int]] = defaultdict(list)

    for element in _sliced_states_iterator(trackers, domain, max_history, tokenizer):
        hashed_state = element.sliced_states_hash
        current_hash = hash(element.event)

        if current_hash not in state_action_mapping[
            hashed_state
        ] or _unlearnable_action(element.event):
            state_action_mapping[hashed_state] += [current_hash]

    # Keep only conflicting `state_action_mapping`s
    # or those mappings that contain `action_unlikely_intent`
    action_unlikely_intent_hash = hash(
        ActionExecuted(action_name=ACTION_UNLIKELY_INTENT_NAME)
    )
    return {
        state_hash: actions
        for (state_hash, actions) in state_action_mapping.items()
        if len(actions) > 1 or action_unlikely_intent_hash in actions
    }


def _unlearnable_action(event: Event) -> bool:
    """Identifies if the action cannot be learned by policies that use story data.

    Args:
        event: An event to be checked.

    Returns:
        `True` if the event can be learned, `False` otherwise.
    """
    return (
        isinstance(event, ActionExecuted)
        and event.action_name == ACTION_UNLIKELY_INTENT_NAME
    )


def _build_conflicts_from_states(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int],
    conflicting_state_action_mapping: Dict[int, List[int]],
    tokenizer: Optional[Tokenizer] = None,
) -> List["StoryConflict"]:
    """Builds a list of `StoryConflict` objects for each given conflict.

    Args:
        trackers: Trackers that contain the states.
        domain: The domain object.
        max_history: Number of turns to take into account for the state descriptions.
        conflicting_state_action_mapping: A dictionary mapping state-hashes to a list
            of actions that follow from each state.
        tokenizer: A tokenizer to tokenize the user messages.

    Returns:
        A list of `StoryConflict` objects that describe inconsistencies in the story
        structure. These objects also contain the history that leads up to the conflict.
    """
    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = {}
    for element in _sliced_states_iterator(trackers, domain, max_history, tokenizer):
        hashed_state = element.sliced_states_hash

        if hashed_state in conflicting_state_action_mapping:
            if hashed_state not in conflicts:
                conflicts[hashed_state] = StoryConflict(element.sliced_states)

            conflicts[hashed_state].add_conflicting_action(
                action=str(element.event), story_name=element.tracker.sender_id
            )

    # Return list of conflicts that arise from unpredictable actions
    # (actions that start the conversation)
    return [
        conflict
        for (hashed_state, conflict) in conflicts.items()
        if conflict.conflict_has_prior_events
    ]


def _sliced_states_iterator(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int],
    tokenizer: Optional[Tokenizer],
) -> Generator[TrackerEventStateTuple, None, None]:
    """Creates an iterator over sliced states.

    Iterate over all given trackers and all sliced states within each tracker,
    where the slicing is based on `max_history`.

    Args:
        trackers: List of trackers.
        domain: Domain (used for tracker.past_states).
        max_history: Assumed `max_history` value for slicing.
        tokenizer: A tokenizer to tokenize the user messages.

    Yields:
        A (tracker, event, sliced_states) triplet.
    """
    for tracker in trackers:
        states = tracker.past_states(domain)

        idx = 0
        for event in tracker.events:
            if isinstance(event, ActionExecuted):
                sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                    states[: idx + 1], max_history
                )
                if tokenizer:
                    _apply_tokenizer_to_states(tokenizer, sliced_states)
                # TODO: deal with oov (different tokens can lead to identical features
                # if some of those tokens are out of vocabulary for all featurizers)
                yield TrackerEventStateTuple(tracker, event, sliced_states)
                idx += 1


def _apply_tokenizer_to_states(tokenizer: Tokenizer, states: List[State]) -> None:
    """Split each user text into tokens and concatenate them again.

    Args:
        tokenizer: A tokenizer to tokenize the user messages.
        states: The states to be tokenized.
    """
    for state in states:
        if USER in state and TEXT in state[USER]:
            state[USER][TEXT] = " ".join(
                token.text
                for token in tokenizer.tokenize(
                    Message({TEXT: state[USER][TEXT]}), TEXT
                )
            )


def _get_previous_event(
    state: Optional[State],
) -> Tuple[Optional[Text], Optional[Text]]:
    """Returns previous event type and name.

    Returns the type and name of the event (action or intent) previous to the
    given state (excluding action_listen).

    Args:
        state: Element of sliced states.

    Returns:
        Tuple of (type, name) strings of the prior event.
    """

    previous_event_type = None
    previous_event_name = None

    # A typical state might be
    # `{'user': {'intent': 'greet'}, 'prev_action': {'action_name': 'action_listen'}}`.
    if not state:
        previous_event_type = None
        previous_event_name = None
    elif (
        PREVIOUS_ACTION in state.keys()
        and "action_name" in state[PREVIOUS_ACTION]
        and state[PREVIOUS_ACTION]["action_name"] != ACTION_LISTEN_NAME
    ):
        previous_event_type = "action"
        previous_event_name = state[PREVIOUS_ACTION]["action_name"]
    elif PREVIOUS_ACTION in state.keys() and "action_text" in state[PREVIOUS_ACTION]:
        previous_event_type = "bot utterance"
        previous_event_name = state[PREVIOUS_ACTION]["action_text"]
    elif USER in state.keys():
        if "intent" in state[USER]:
            previous_event_type = "intent"
            previous_event_name = state[USER]["intent"]
        elif "text" in state[USER]:
            previous_event_type = "user utterance"
            previous_event_name = state[USER]["text"]

    if not isinstance(previous_event_name, (str, type(None))):
        # While the Substate type doesn't restrict the value of `action_text` /
        # `intent`, etc. to be a string, it always should be
        raise TypeError(
            f"The value '{previous_event_name}' in the substate should be a string or "
            f"None, not {type(previous_event_name)}. Did you modify Rasa source code?"
        )

    return previous_event_type, previous_event_name
