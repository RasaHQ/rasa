from collections import defaultdict
import logging
from typing import Dict, Generator, List, NamedTuple, Optional, Text, Tuple, Any

from rasa.core.featurizers.tracker_featurizers import MaxHistoryTrackerFeaturizer
from rasa.shared.core.constants import ACTION_LISTEN_NAME, PREVIOUS_ACTION, USER
from rasa.shared.core.domain import Domain, PREV_PREFIX, State, SubState
from rasa.shared.core.events import ActionExecuted, Event
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.nlu.constants import INTENT

from rasa.nlu.model import Trainer
from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.utils.io import raise_warning

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
        return hash(_as_sorted_text(self.sliced_states))


def _as_sorted_text(obj: Any) -> Text:
    """Returns the string of `obj` after sorting lists and dicts.

    Args:
        obj: Something made up of lists and dicts and stringifiable objects.

    Returns:
        A string representation of the object that doesn't change
        randomly due to unsorted dicts or sets.
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return str(
            [
                (_as_sorted_text(key), _as_sorted_text(value))
                for key, value in sorted(obj.items())
            ]
        )
    elif isinstance(obj, (list, set)):
        return str(sorted([_as_sorted_text(element) for element in obj]))
    return str(obj)


def _get_length_of_longest_story(
    trackers: List[TrackerWithCachedStates], domain: Domain
) -> int:
    """Returns the longest story in the given trackers.

    Args:
        trackers: Trackers to get stories from.
        domain: The domain.

    Returns:
        The maximal length of any story
    """
    return max([len(tracker.past_states(domain)) for tracker in trackers])


def find_story_conflicts(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int] = None,
    nlu_config: Optional[RasaNLUModelConfig] = None,
) -> List[StoryConflict]:
    """Generates `StoryConflict` objects, describing conflicts in the given trackers.

    Args:
        trackers: Trackers in which to search for conflicts.
        domain: The domain.
        max_history: The maximum history length to be taken into account.
        nlu_config: NLU config.

    Returns:
        StoryConflict objects.
    """
    if max_history:
        logger.info(
            f"Considering the preceding {max_history} turns for conflict analysis."
        )
    else:
        logger.info("Considering all preceding turns for conflict analysis.")

    tokenizing_function = _get_tokenizing_function_from_nlu_config(nlu_config)

    # We do this in two steps, to reduce memory consumption:

    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    conflicting_state_action_mapping = _find_conflicting_states(
        trackers, domain, max_history, tokenizing_function
    )

    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = _build_conflicts_from_states(
        trackers,
        domain,
        max_history,
        conflicting_state_action_mapping,
        tokenizing_function,
    )

    return conflicts


def _get_tokenizing_function_from_nlu_config(
    nlu_config: Optional[RasaNLUModelConfig] = None,
) -> Optional[callable]:
    """Extracts the `tokenize` function of the first Tokenizer in the pipeline.

    Args:
        nlu_config: NLU Config.
    """
    if not nlu_config:
        return None

    pipeline: List[Component] = Trainer(
        nlu_config, skip_validation=True
    ).pipeline  # ToDo: ComponentBuilder?
    tokenizer: Optional[Tokenizer] = None
    for component in pipeline:
        if isinstance(component, Tokenizer) and tokenizer:
            raise_warning(
                "The pipeline contains more than one tokenizer. "
                "Only the first tokenizer will be used for story validation.",
                category=UserWarning,
            )
        elif isinstance(component, Tokenizer):
            tokenizer = component

    return tokenizer.tokenize if tokenizer else None


def _find_conflicting_states(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int],
    tokenizing_function: Optional[callable],
) -> Dict[int, Optional[List[Text]]]:
    """Identifies all states from which different actions follow.

    Args:
        trackers: Trackers that contain the states.
        domain: The domain object.
        max_history: Number of turns to take into account for the state descriptions.
        tokenizing_function: A `Tokenizer.tokenize` function.

    Returns:
        A dictionary mapping state-hashes to a list of actions that follow from each state.
    """
    # Create a 'state -> list of actions' dict, where the state is
    # represented by its hash
    state_action_mapping = defaultdict(list)
    for element in _sliced_states_iterator(
        trackers, domain, max_history, tokenizing_function
    ):
        hashed_state = element.sliced_states_hash
        current_hash = hash(element.event)
        if current_hash not in state_action_mapping[hashed_state]:
            state_action_mapping[hashed_state] += [current_hash]

    # Keep only conflicting `state_action_mapping`s
    return {
        state_hash: actions
        for (state_hash, actions) in state_action_mapping.items()
        if len(actions) > 1
    }


def _build_conflicts_from_states(
    trackers: List[TrackerWithCachedStates],
    domain: Domain,
    max_history: Optional[int],
    conflicting_state_action_mapping: Dict[int, Optional[List[Text]]],
    tokenizing_function: Optional[callable],
) -> List["StoryConflict"]:
    """Builds a list of `StoryConflict` objects for each given conflict.

    Args:
        trackers: Trackers that contain the states.
        domain: The domain object.
        max_history: Number of turns to take into account for the state descriptions.
        conflicting_state_action_mapping: A dictionary mapping state-hashes to a list of actions
                                          that follow from each state.
        tokenizing_function: A `Tokenizer.tokenize` function.

    Returns:
        A list of `StoryConflict` objects that describe inconsistencies in the story
        structure. These objects also contain the history that leads up to the conflict.
    """
    # Iterate once more over all states and note the (unhashed) state,
    # for which a conflict occurs
    conflicts = {}
    for element in _sliced_states_iterator(
        trackers, domain, max_history, tokenizing_function
    ):
        hashed_state = element.sliced_states_hash

        if hashed_state in conflicting_state_action_mapping:
            if hashed_state not in conflicts:
                conflicts[hashed_state] = StoryConflict(element.sliced_states)

            conflicts[hashed_state].add_conflicting_action(
                action=str(element.event), story_name=element.tracker.sender_id,
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
    tokenizing_function: Optional[callable],
) -> Generator[TrackerEventStateTuple, None, None]:
    """Creates an iterator over sliced states.

    Iterate over all given trackers and all sliced states within each tracker,
    where the slicing is based on `max_history`.

    Args:
        trackers: List of trackers.
        domain: Domain (used for tracker.past_states).
        max_history: Assumed `max_history` value for slicing.
        tokenizing_function: A `Tokenizer.tokenize` function.

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
                if tokenizing_function:
                    _apply_tokenizer_to_states(tokenizing_function, sliced_states)
                # ToDo: deal with oov (different tokens can lead to identical features if some of those tokens are out of vocabulary for all featurizers)
                yield TrackerEventStateTuple(tracker, event, sliced_states)
                idx += 1


def _apply_tokenizer_to_states(
    tokenizing_function: callable, states: List[State]
) -> None:
    """Split each user text into tokens and concatenate them again.

    Args:
        tokenizing_function: Should take a message and an attribute and return the tokens,
        just like `Tokenizer.tokenize`.
        states: The states to be tokenized.
    """
    for state in states:
        if USER in state:
            state[USER][TEXT] = "".join(
                token.text
                for token in tokenizing_function(
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
        # While the Substate type doesn't restrict the value of `action_text` / `intent`, etc. to be a string, it always should be
        raise TypeError(
            f"The value '{previous_event_name}' in the substate should be a string or None, not {type(previous_event_name)}. Did you modify Rasa source code?"
        )

    return previous_event_type, previous_event_name
