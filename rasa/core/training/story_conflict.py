from collections import defaultdict, namedtuple
from typing import List, Optional, Dict, Text, Tuple, Generator, NamedTuple

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain
from rasa.core.events import ActionExecuted, Event
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.nlu.constants import INTENT_ATTRIBUTE
from rasa.core.training.generator import TrackerWithCachedStates


class StoryConflict:
    def __init__(self, sliced_states: List[Optional[Dict[Text, float]]],) -> None:
        self._sliced_states = sliced_states
        self._conflicting_actions = defaultdict(
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
                f"CONFLICT after {last_event_type} '{last_event_name}':\n"
            )
        else:
            conflict_message = f"CONFLICT at the beginning of stories:\n"

        # List which stories are in conflict with one another
        for action, stories in self._conflicting_actions.items():
            conflict_message += "  " + self._summarize_conflict(action, stories)

        return conflict_message

    @staticmethod
    def _summarize_conflict(action: Text, stories: List[Text]) -> Text:
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
            raise ValueError("Trying to summarize conflict without stories.")

        return f"{action} predicted in {conflict_description}\n"


class TrackerEventStateTuple(NamedTuple):
    """Holds a tracker, an event, and sliced states associated with those."""

    tracker: TrackerWithCachedStates
    event: Event
    sliced_states: List[Dict[Text, float]]

    @property
    def sliced_states_hash(self) -> int:
        return hash(str(list(self.sliced_states)))


def find_story_conflicts(
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> List[StoryConflict]:
    """Generates a list of `StoryConflict` objects, describing conflicts in the given trackers.

    Args:
        trackers: Trackers in which to search for conflicts.
        domain: The domain.
        max_history: The maximum history length to be taken into account.
    Returns:
        List of conflicts.
    """
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
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> Dict[int, Optional[List[Text]]]:
    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    state_action_mapping = defaultdict(list)
    for element in _sliced_states_iterator(trackers, domain, max_history):
        hashed_state = element.sliced_states_hash
        if element.event.as_story_string() not in state_action_mapping[hashed_state]:
            state_action_mapping[hashed_state] += [element.event.as_story_string()]

    # Keep only conflicting `state_action_mapping`s
    return {
        state_hash: actions
        for (state_hash, actions) in state_action_mapping.items()
        if len(actions) > 1
    }


def _build_conflicts_from_states(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: int,
    conflicting_state_action_mapping: Dict[int, Optional[List[Text]]],
) -> List["StoryConflict"]:
    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = {}
    for element in _sliced_states_iterator(trackers, domain, max_history):
        hashed_state = element.sliced_states_hash

        if hashed_state in conflicting_state_action_mapping:
            if hashed_state not in conflicts:
                conflicts[hashed_state] = StoryConflict(element.sliced_states)

            conflicts[hashed_state].add_conflicting_action(
                action=element.event.as_story_string(),
                story_name=element.tracker.sender_id,
            )

    # Return list of conflicts that arise from unpredictable actions
    # (actions that start the conversation)
    return [
        conflict
        for (hashed_state, conflict) in conflicts.items()
        if conflict.conflict_has_prior_events
    ]


def _sliced_states_iterator(
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> Generator[TrackerEventStateTuple, None, None]:
    """Creates an iterator over sliced states.

    Iterate over all given trackers and all sliced states within each tracker,
    where the slicing is based on `max_history`.

    Args:
        trackers: List of trackers.
        domain: Domain (used for tracker.past_states).
        max_history: Assumed `max_history` value for slicing.

    Yields:
        A (tracker, event, sliced_states) triplet.
    """
    for tracker in trackers:
        states = tracker.past_states(domain)
        states = [dict(state) for state in states]

        idx = 0
        for event in tracker.events:
            if isinstance(event, ActionExecuted):
                sliced_states = MaxHistoryTrackerFeaturizer.slice_state_history(
                    states[: idx + 1], max_history
                )
                yield TrackerEventStateTuple(tracker, event, sliced_states)
                idx += 1


def _get_previous_event(
    state: Optional[Dict[Text, float]]
) -> Tuple[Optional[Text], Optional[Text]]:
    """Returns previous event type and name.

    Returns the type and name of the event (action or intent) previous to the
    given state.

    Args:
        state: Element of sliced states.

    Returns:
        Tuple of (type, name) strings of the prior event.
    """

    previous_event_type = None
    previous_event_name = None

    if not state:
        return previous_event_type, previous_event_name

    # A typical state is, for example,
    # `{'prev_action_listen': 1.0, 'intent_greet': 1.0, 'slot_cuisine_0': 1.0}`.
    # We need to look out for `prev_` and `intent_` prefixes in the labels.
    for turn_label in state:
        if (
            turn_label.startswith(PREV_PREFIX)
            and turn_label.replace(PREV_PREFIX, "") != ACTION_LISTEN_NAME
        ):
            # The `prev_...` was an action that was NOT `action_listen`
            return "action", turn_label.replace(PREV_PREFIX, "")
        elif turn_label.startswith(INTENT_ATTRIBUTE + "_"):
            # We found an intent, but it is only the previous event if
            # the `prev_...` was `prev_action_listen`, so we don't return.
            previous_event_type = "intent"
            previous_event_name = turn_label.replace(INTENT_ATTRIBUTE + "_", "")

    return previous_event_type, previous_event_name
