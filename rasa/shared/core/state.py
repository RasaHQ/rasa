"""TODO/FIXME: temporary workaround until we introduce a state interface"""
from typing import Optional, List, Text
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, LOOP_NAME
from rasa.shared.core.domain import State, SubState
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, USER, TEXT
from rasa.shared.core.trackers import is_prev_action_listen_in_state
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    LOOP_NAME,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    ENTITIES,
    ACTION_TEXT,
)

# TODO: move all state (not tracker) related code from domain here...

# # State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# # representing the origin of a SubState;
# # the values are SubStates, that contain the information needed for featurization
# SubState = Dict[Text, Union[Text, Tuple[Union[float, Text]]]]
# State = Dict[Text, SubState]


def create_substate_from_action(action: Text, as_text: bool = False) -> SubState:
    key = ACTION_TEXT if as_text else ACTION_NAME
    return {key: action}


def get_intent(state: State) -> Optional[Text]:
    return state.get(USER, {}).get(TEXT)


def get_active_loop_name(state: State) -> Optional[Text]:
    """
    TODO: docstr pointing out that "should not be set" won't be returned
    """
    if (
        not state.get(ACTIVE_LOOP)
        or state[ACTIVE_LOOP].get(LOOP_NAME) == SHOULD_NOT_BE_SET
    ):
        return None
    return state[ACTIVE_LOOP].get(LOOP_NAME)


def get_previous_action(state: State) -> Optional[Text]:
    return state.get(PREVIOUS_ACTION, {}).get(ACTION_NAME)


def previous_action_was_listen(state: State) -> bool:
    """Check if action_listen is the previous executed action.

    Args:
        state: The state for which the check should be performed

    Return:
        boolean value indicating whether action_listen is previous action
    """
    return get_previous_action(state) == ACTION_LISTEN_NAME


def forget_slots(state: State, slots: Optional[List[Text]]) -> None:
    slots = slots or []
    for slot in slots:
        state.get(SLOTS, {}).pop(slot, None)


def forget_active_loop_if_name_is_in(
    state: State, loop_names: Optional[List[Text]]
) -> None:
    """Only forgets about hte acti"""
    # remove active loop which only occur in rules but not in stories
    loop_names = loop_names or []
    if state.get(ACTIVE_LOOP, {}).get(LOOP_NAME) in loop_names:
        del state[ACTIVE_LOOP]


def forget_user_text(state: State) -> None:
    if state.get(USER, {}).get(INTENT):
        del state[USER][TEXT]


def forget_states_after_last_user_input(  # FIXME: meaningful name?
    states: List[State], use_text_for_last_user_input: bool
) -> None:
    """

    # TODO: docstr

    Modifies the given list of states in-place.
    """
    last_state = states[-1]
    # only update the state of the real user utterance
    if not is_prev_action_listen_in_state(last_state):
        return

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

    # "remove user text if intent"
    for state in states:
        # remove text features to only use intent
        if state.get(USER, {}).get(INTENT) and state.get(USER, {}).get(TEXT):
            del state[USER][TEXT]
