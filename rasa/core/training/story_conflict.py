from collections import defaultdict
from typing import List, Optional, Dict, Text

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import PREV_PREFIX, Domain
from rasa.core.events import ActionExecuted, Event
from rasa.core.featurizers import MaxHistoryTrackerFeaturizer
from rasa.nlu.constants import MESSAGE_INTENT_ATTRIBUTE
from rasa.core.training.generator import TrackerWithCachedStates


class StoryConflict:
    def __init__(
        self, sliced_states: List[Optional[Dict[Text, float]]],
    ) -> None:
        self.sliced_states = sliced_states
        self._conflicting_actions = defaultdict(list)  # {"action": ["story_1", ...], ...}
        self.correct_response = None

    def __hash__(self) -> int:
        return hash(str(list(self.sliced_states)))

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
    def has_prior_events(self) -> bool:
        """Checks if prior events exist.

        Returns:
            True if anything has happened before this conflict, otherwise False.
        """
        return _get_prev_event(self.sliced_states[-1])[0] is not None

    def __str__(self) -> Text:
        # Describe where the conflict occurs in the stories
        last_event_type, last_event_name = _get_prev_event(self.sliced_states[-1])
        if last_event_type:
            conflict_string = f"CONFLICT after {last_event_type} '{last_event_name}':\n"
        else:
            conflict_string = f"CONFLICT at the beginning of stories:\n"

        # List which stories are in conflict with one another
        for action, stories in self._conflicting_actions.items():
            # Summarize if necessary
            story_desc = {
                1: "'{}'",
                2: "'{}' and '{}'",
                3: "'{}', '{}', and '{}'",
            }.get(len(stories))
            if story_desc:
                story_desc = story_desc.format(*stories)
            else:
                # Four or more stories are present
                story_desc = f"'{stories[0]}' and {len(stories) - 1} other trackers"

            conflict_string += f"  {action} predicted in {story_desc}\n"

        return conflict_string


def find_story_conflicts(
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> List[StoryConflict]:
    """Generates a list of StoryConflict objects, describing conflicts in the given trackers.

    Args:
        trackers: Trackers in which to search for conflicts
        domain: The domain
        max_history: The maximum history length to be taken into account
    Returns:
        List of conflicts
    """

    # We do this in two steps, to reduce memory consumption:

    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    state_action_dict = _find_conflicting_states(trackers, domain, max_history)

    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = _build_conflicts_from_states(trackers, domain, max_history, state_action_dict)

    return conflicts


def _find_conflicting_states(
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> Dict[int, Optional[List[Text]]]:
    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    state_action_dict = defaultdict(list)
    for tracker, event, sliced_states in _sliced_states_iterator(
        trackers, domain, max_history
    ):
        hashed_state = hash(str(list(sliced_states)))
        if event.as_story_string() not in state_action_dict[hashed_state]:
            state_action_dict[hashed_state] += [event.as_story_string()]

    # Keep only conflicting `state_action_dict`s
    return {
        state_hash: actions
        for (state_hash, actions) in state_action_dict.items()
        if len(actions) > 1
    }


def _build_conflicts_from_states(
    trackers: List["TrackerWithCachedStates"],
    domain: Domain,
    max_history: int,
    rules: Dict[int, Optional[List[Text]]],
) -> List["StoryConflict"]:
    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = {}
    for tracker, event, sliced_states in _sliced_states_iterator(
        trackers, domain, max_history
    ):
        hashed_state = hash(str(list(sliced_states)))

        if hashed_state in rules and hashed_state not in conflicts:
            conflicts[hashed_state] = StoryConflict(sliced_states)

        if hashed_state in rules:
            conflicts[hashed_state].add_conflicting_action(
                action=event.as_story_string(), story_name=tracker.sender_id
            )

    # Remove conflicts that arise from unpredictable actions
    return [conflict for (hashed_state, conflict) in conflicts.items() if conflict.has_prior_events]


def _sliced_states_iterator(
    trackers: List[TrackerWithCachedStates], domain: Domain, max_history: int
) -> (TrackerWithCachedStates, Event, List[Dict[Text, float]]):
    """Creates an iterator over sliced states.

    Iterate over all given trackers and all sliced states within each tracker,
    where the slicing is based on `max_history`.

    Args:
        trackers: List of trackers
        domain: Domain (used for tracker.past_states)
        max_history: Assumed `max_history` value for slicing

    Yields:
        A (tracker, event, sliced_states) triplet
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
                yield tracker, event, sliced_states
                idx += 1


def _get_prev_event(
    state: Optional[Dict[Text, float]]
) -> [Optional[Text], Optional[Text]]:
    """Returns previous event type and name.

    Returns the type and name of the event (action or intent) previous to the
    given state.

    Args:
        state: Element of sliced states.

    Returns:
        Tuple of (type, name) strings of the prior event.
    """
    prev_event_type = None
    prev_event_name = None

    if not state:
        return prev_event_type, prev_event_name

    for k in state:
        if k.startswith(PREV_PREFIX) and k[len(PREV_PREFIX) :] != ACTION_LISTEN_NAME:
            prev_event_type = "action"
            prev_event_name = k[len(PREV_PREFIX) :]

        if not prev_event_type and k.startswith(MESSAGE_INTENT_ATTRIBUTE + "_"):
            prev_event_type = "intent"
            prev_event_name = k[len(MESSAGE_INTENT_ATTRIBUTE + "_") :]

    return prev_event_type, prev_event_name
