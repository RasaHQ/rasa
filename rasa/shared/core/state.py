from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Text, Union, Tuple

from rasa.shared.nlu.training_data.features import Features
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, LOOP_NAME
from rasa.shared.core.domain import State, SubState
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, USER, TEXT
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    FrozenState,
    is_prev_action_listen_in_state,
)
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    LOOP_NAME,
    SHOULD_NOT_BE_SET,
    PREVIOUS_ACTION,
    ACTIVE_LOOP,
    ENTITIES,
    ACTION_TEXT,
    ACTION_UNLIKELY_INTENT_NAME,
)

# TODO: move all state (not tracker) related code from domain here...

# # State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# # representing the origin of a SubState;
# # the values are SubStates, that contain the information needed for featurization
# SubState = Dict[Text, Union[Text, Tuple[Union[float, Text]]]]
# State = Dict[Text, SubState]


######## Message / State


from abc import ABC
from typing import TypeVar, Generic

T = TypeVar("T")


@dataclass
class Information(Generic[T], ABC):
    raw: T
    featurized: List[Features]


class DisplayedText(Information[Text]):
    pass


class Action(Information[Text]):
    text: DisplayedText
    pass


class Intents(Information[List[Text]]):
    pass


class Entities(Information[List[Text]]):
    pass


class MetaData(Information):
    pass


########## Creation

# FIXME: move all here...


def get_active_state(self, omit_unset_slots: bool = False,) -> State:
    """Given a dialogue tracker, makes a representation of current dialogue state.

    Args:
        tracker: dialog state tracker containing the dialog so far
        omit_unset_slots: If `True` do not include the initial values of slots.

    Returns:
        A representation of the dialogue's current state.
    """
    state = {
        USER: self._get_user_sub_state(),
        SLOTS: self._get_slots_sub_state(omit_unset_slots=omit_unset_slots),
        PREVIOUS_ACTION: self.latest_action,
        ACTIVE_LOOP: self._get_active_loop_sub_state(),
    }
    return self._clean_state(state)


def get_slots_sub_state(
    self, omit_unset_slots: bool = False,
) -> Dict[Text, Union[Text, Tuple[float]]]:
    """Sets all set slots with the featurization of the stored value.

    Args:
        tracker: dialog state tracker containing the dialog so far
        omit_unset_slots: If `True` do not include the initial values of slots.

    Returns:
        a dictionary mapping slot names to their featurization
    """
    slots = {}
    for slot_name, slot in self.slots.items():
        if slot is not None and slot.as_feature():
            if omit_unset_slots and not slot.has_been_set:
                continue
            if slot.value == SHOULD_NOT_BE_SET:
                slots[slot_name] = SHOULD_NOT_BE_SET
            elif any(slot.as_feature()):
                # only add slot if some of the features are not zero
                slots[slot_name] = tuple(slot.as_feature())

    return slots


def get_active_loop_sub_state(self, tracker: DialogueStateTracker) -> Dict[Text, Text]:
    """Turn tracker's active loop into a state name.
    Args:
        tracker: dialog state tracker containing the dialog so far
    Returns:
        a dictionary mapping "name" to active loop name if present
    """

    # we don't use tracker.active_loop_name
    # because we need to keep should_not_be_set
    active_loop: Optional[Text] = self.active_loop.get(LOOP_NAME)
    if active_loop:
        return {LOOP_NAME: active_loop}
    else:
        return {}


def clean_state(state: State) -> State:
    return {
        state_type: sub_state for state_type, sub_state in state.items() if sub_state
    }


def past_states(
    self,
    omit_unset_slots: bool = False,
    ignore_rule_only_turns: bool = False,
    ignored_active_loop_names: Optional[List[Text]] = None,
    ignored_slots: Optional[List[Text]] = None,
) -> List[State]:
    """Generates the past states of this tracker based on the history.

    Args:
        domain: The Domain.
        omit_unset_slots: If `True` do not include the initial values of slots.
        ignore_rule_only_turns: If True ignore dialogue turns that are present
            only in rules.
        ignored_slots: slots to be ignored iff `ignore_rule_only_turns` is True
        ignored_active_loop_names: active loops with names included in this
            list will be ignored iff `ignore_rule_only_turns` is True

    Returns:
        A list of states
    """
    states = []
    last_ml_action_sub_state = None
    turn_was_hidden = False
    for tr, hide_rule_turn in self.generate_all_prior_trackers():
        if ignore_rule_only_turns:
            # remember previous ml action based on the last non hidden turn
            # we need this to override previous action in the ml state
            if not turn_was_hidden:
                last_ml_action_sub_state = self.latest_action

            # followup action or happy path loop prediction
            # don't change the fact whether dialogue turn should be hidden
            if (
                not tr.followup_action
                and not tr.latest_action_name == tr.active_loop_name
            ):
                turn_was_hidden = hide_rule_turn

            if turn_was_hidden:
                continue

        state = self.get_active_state(tr, omit_unset_slots=omit_unset_slots)

        if ignore_rule_only_turns:
            # clean state from only rule features
            # TODO: generic state.forget substates... (?)
            forget_active_loop_if_name_is_in(ignored_active_loop_names)
            forget_slots(ignored_slots)
            # make sure user input is the same as for previous state
            # for non action_listen turns
            if states:
                self._substitute_rule_only_user_input(state, states[-1])
            # substitute previous rule action with last_ml_action_sub_state
            if last_ml_action_sub_state:
                state[PREVIOUS_ACTION] = last_ml_action_sub_state

        states.append(self._clean_state(state))

    return states


def get_prev_action_sub_state(tracker: DialogueStateTracker) -> Dict[Text, Text]:
    """Turn the previous taken action into a state name.
    Args:
        tracker: dialog state tracker containing the dialog so far
    Returns:
        a dictionary with the information on latest action
    """
    return tracker.latest_action


def create_action_sub_state(action: Text, as_text: bool = False) -> SubState:
    key = ACTION_TEXT if as_text else ACTION_NAME
    return {key: action}


########## Helper


def freeze(state: State) -> FrozenState:
    """Convert State dict into a hashable format FrozenState.

    Args:
        state: The state which should be converted

    Return:
        hashable form of the state of type `FrozenState`
    """
    return frozenset(
        {
            key: frozenset(values.items())
            if isinstance(values, Dict)
            else frozenset(values)
            for key, values in state.items()
        }.items()
    )


def copy(state: State, key_dict: Dict[Text, Optional[List[Text]]]):
    """Copy only the requested (state_type, attribute) entries, if they exist.

    Args:
        key_dict: mapping substate_types to a list of attribute names
    """
    state_copy = dict()
    for key, sub_keys in key_dict.items():
        sub_state = state.get(key, None)
        if sub_state is not None:
            if sub_keys is None:
                sub_state_copy = copy.deep_copy(sub_state)
            else:
                sub_state_copy = {}
                for sub_key in sub_keys:
                    item = sub_state.get(sub_key, None)
                    if item is not None:
                        sub_state_copy = copy.deepcopy(item)
            state_copy[key] = sub_state_copy
    return state_copy


########## Get Attributes


def get_user_text(state: State) -> Optional[Text]:
    return state.get(USER, {}).get(TEXT)


def get_user_intent(state: State) -> Optional[Text]:
    return state.get(USER, {}).get(INTENT)


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


########## Check Attributes


def previous_action_was_unlikely_intent_action(state: State) -> bool:
    return get_previous_action() == ACTION_UNLIKELY_INTENT_NAME


def previous_action_was_listen(state: State) -> bool:
    """Check if action_listen is the previous executed action.

    Args:
        state: The state for which the check should be performed

    Return:
        boolean value indicating whether action_listen is previous action
    """
    return get_previous_action(state) == ACTION_LISTEN_NAME


########## Manipulate


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
    state[USER].pop(TEXT, None)


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
