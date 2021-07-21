from typing import Any, Dict, Optional, List, Text, Union, Tuple, Sized, Set, cast

from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, LOOP_NAME
from rasa.shared.core.domain import Domain, State, SubState
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, USER, TEXT
from rasa.shared.core.trackers import (
    DialogueStateTracker,
    FrozenState,
    is_prev_action_listen_in_state,
)
from rasa.shared.core import constants as shared_core_constants
from rasa.shared.nlu import constants as shared_nlu_constants
from rasa.shared.core.events import ActionExecuted, UserUttered


########################################################################################
#                               Definition
########################################################################################


# # State is a dictionary with keys (USER, PREVIOUS_ACTION, SLOTS, ACTIVE_LOOP)
# # representing the origin of a SubState;
# # the values are SubStates, that contain the information needed for featurization
# SubState = Dict[Text, Union[Text, Tuple[Union[float, Text]]]]
# State = Dict[Text, SubState]

# class State:
#     user : Dict
#     slots : Dict
#     previous_action : Dict[Text, Text]
#     active_loop : Dict[Text, Text]

########################################################################################
#                               Helper Functions
########################################################################################


def shallow_copy_without_empty_values(
    given: Dict[Any, Optional[Sized]]
) -> Dict[Any, Optional[Sized]]:
    """
    NOTE: this was _clean_state (with the difference that that would've dropped values
    evaluating to False (of which we have nothing yet))
    """
    return {key: val for key, val in given.items() if val is None or not len(val) == 0}


########################################################################################
#                               SubState Creation
########################################################################################

# TODO: rename these get-functions


def get_active_state(
    tracker: DialogueStateTracker, domain: Domain, omit_unset_slots: bool = False,
) -> State:
    """Given a dialogue tracker, makes a representation of current dialogue state.

    Args:
        tracker: dialog state tracker containing the dialog so far
        omit_unset_slots: If `True` do not include the initial values of slots.

    Returns:
        A representation of the dialogue's current state.
    """
    state = {
        shared_nlu_constants.USER: get_user_sub_state(
            user_uttered_event=tracker.latest_message,
            apply_restriction_from_intent=True,
            domain=domain,
        ),
        shared_nlu_constants.SLOTS: get_slots_sub_state(
            tracker, omit_unset_slots=omit_unset_slots
        ),
        shared_nlu_constants.PREVIOUS_ACTION: get_prev_action_sub_state(
            tracker.latest_action
        ),
        # == get_prev_action_sub_state(tracker)
        # == {ACTION_NAME: <Text>} or {ACTION_TEXT : <Text>}
        shared_nlu_constants.ACTIVE_LOOP: get_active_loop_sub_state(tracker),
        # == <TEXT>
    }
    return shallow_copy_without_empty_values(state)


def get_user_sub_state(
    user_uttered_event: UserUttered,
    apply_restriction_from_intent: bool = False,
    domain: Optional[Domain] = None,
) -> Dict[Text, Union[Text, Tuple[Text]]]:
    """Turns latest UserUttered event into a substate.

    The substate will contain intent, text, and entities (if any are present).

    Args:
        tracker: dialog state tracker containing the dialog so far
    Returns:
        a dictionary containing intent, text and set entities

    TODO: why does *the event* get to decide how it is featurized? and why is that
      ignored in UserUttered.as_sub_state ?
    NOTE: to replicate UserUttered.as_sub_state() call this function with
      apply_restriction_from_intent set to False (and no domain, won't be used)
    """
    if not user_uttered_event or user_uttered_event.is_empty():
        # where empty means: no intent, no text and no entities
        return dict()

    # TODO: Why is this defined in event and how is it set?
    # During training we expect either intent_name or text to be set
    #   -> means use_text_for_featurization should be True/False (?)
    # During prediction both will be set
    #   -> means use_text_for_featurization should be None (?)

    featurization_undefined = user_uttered_event.use_text_for_featurization is None
    featurization_using_text = user_uttered_event.use_text_for_featurization

    sub_state = {}
    if featurization_undefined or featurization_using_text:
        sub_state[shared_nlu_constants.TEXT] = user_uttered_event.text
    if (
        featurization_undefined
        or not featurization_using_text
        or apply_restriction_from_intent
    ):
        intent_name = user_uttered_event.intent_name
        # don't add entities for e2e utterances # TODO: ? / why defined in event?
        entity_names = collect_all_entity_type_role_group_combinations(
            user_uttered_event.entities
        )

        if apply_restriction_from_intent:
            # FIXME: here, we don't care whether the featurization method was undefined
            # or whether we don't "use_text_for_featurization" ... which matches what is
            # happening in the current version. But why? :)

            if domain is None:
                raise ValueError(
                    "Expected domain to be able to filter entities according to "
                    "restrictions defined in intent configuration."
                )

            # Filter by intent config
            intent_config = domain.intent_config(intent_name)
            restriction = intent_config.get(shared_nlu_constants.USED_ENTITIES_KEY)

            if restriction:
                entity_names = entity_names.intersection(restriction)

                # Sort entities so that any derived state representation is consistent
                # acrossruns and invariant to the order in which the entities for an
                # utterance are listed in data files.
                entity_names = tuple(sorted(entity_names))

        sub_state[shared_nlu_constants.ENTITIES] = entity_names
        sub_state[shared_nlu_constants.INTENT] = intent_name

    return shallow_copy_without_empty_values(sub_state)


def collect_all_entity_type_role_group_combinations(entities: List[Dict[Text, Text]]):
    """Generates all combinations of entity type, role and group we're interested in.

    That is, an entity type can appear on it's own, in conjunction with a role tag or
    in conjunction with a group tag.

    NOTE: this was
    - part of as_sub_state in UserUtteredEvent and
    - part of _get_featurized_entities of domain

    # FIXME: can ENTITY_ATTRIBUTE_TYPE be NONE?
    # This was (half-way becuase of None+role/type combinations) filtered out
    # in _get_featurize but not in as_sub_state ... ?
    """
    combinations = [
        entity.get(shared_nlu_constants.ENTITY_ATTRIBUTE_TYPE) for entity in entities
    ]
    combinations.extend(
        (
            f"{entity.get(shared_nlu_constants.ENTITY_ATTRIBUTE_TYPE)}"
            f"{shared_nlu_constants.ENTITY_LABEL_SEPARATOR}"
            f"{entity.get(shared_nlu_constants.ENTITY_ATTRIBUTE_ROLE)}"
        )
        for entity in entities
        if shared_nlu_constants.ENTITY_ATTRIBUTE_ROLE in entity
    )
    combinations.extend(
        (
            f"{entity.get(shared_nlu_constants.ENTITY_ATTRIBUTE_TYPE)}"
            f"{shared_nlu_constants.ENTITY_LABEL_SEPARATOR}"
            f"{entity.get(shared_nlu_constants.ENTITY_ATTRIBUTE_GROUP)}"
        )
        for entity in entities
        if shared_nlu_constants.ENTITY_ATTRIBUTE_GROUP in entity
    )
    return combinations


def get_slots_sub_state(
    tracker, omit_unset_slots: bool = False,
) -> Dict[Text, Union[Text, Tuple[float]]]:
    """Sets all set slots with the featurization of the stored value.

    Args:
        tracker: dialog state tracker containing the dialog so far
        omit_unset_slots: If `True` do not include the initial values of slots.

    Returns:
        a dictionary mapping slot names to their featurization
    """
    slots = {}
    for slot_name, slot in tracker.slots.items():
        if (
            slot is not None
            and slot.as_feature()  # TODO: "as_feature"? This is not a `Features`
            and not (omit_unset_slots and not slot.has_been_set)
        ):
            if slot.value == shared_nlu_constants.SHOULD_NOT_BE_SET:
                slots[slot_name] = shared_nlu_constants.SHOULD_NOT_BE_SET
            elif any(slot.as_feature()):
                # only add slot if some of the features are not zero
                slots[slot_name] = tuple(slot.as_feature())

    return slots


def get_active_loop_sub_state(tracker: DialogueStateTracker) -> Dict[Text, Text]:
    """Turn tracker's active loop into a state name.
    Args:
        tracker: dialog state tracker containing the dialog so far
    Returns:
        a dictionary mapping "name" to active loop name if present
    """
    # we don't use tracker.active_loop_name
    # because we need to keep should_not_be_set
    active_loop: Optional[Text] = tracker.active_loop.get(LOOP_NAME)
    if active_loop:
        return {LOOP_NAME: active_loop}
    else:
        return {}


def get_prev_action_sub_state(action_executed: ActionExecuted) -> Dict[Text, Text]:
    if action_executed.action_name:
        key = shared_nlu_constants.ACTION_NAME
        value = action_executed.action_name
    else:
        key = shared_nlu_constants.ACTION_TEXT
        value = cast(Text, action_executed.action_text)
    return {key: value}


# def create_action_sub_state(text: Text, as_text: bool = False) -> SubState:
#     key = shared_nlu_constants.ACTION_TEXT if as_text else ACTION_NAME
#     return {key: text}


########################################################################################
#                               State Helper Functions
########################################################################################


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


def get_schema(state: State):
    """
    """
    schema = dict()
    for key, sub_keys in state.items():
        schema.setdefault(key, sub_keys)
    return schema


def copy(state: State, key_dict: Dict[Text, Optional[List[Text]]]):
    """Copy only the requested (state_type, attribute) entries, if they exist.

    # TODO: rename to deep_copy or convert to shallow copy (is deep copy needed?)

    Args:
        key_dict: mapping substate_types to a list of attribute names
    Returns:
        a deep copy of the given state that only contains those (substate_type,
        attribute) combinations that are needed.
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


########################################################################################
#                               Getter
########################################################################################


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
        or state[ACTIVE_LOOP].get(LOOP_NAME) == shared_nlu_constants.SHOULD_NOT_BE_SET
    ):
        return None
    return state[ACTIVE_LOOP].get(LOOP_NAME)


def get_previous_action(state: State) -> Optional[Text]:
    return state.get(shared_nlu_constants.PREVIOUS_ACTION, {}).get(ACTION_NAME)


########################################################################################
#                              Checks
########################################################################################


def previous_action_was_unlikely_intent_action(state: State) -> bool:
    return (
        get_previous_action(state) == shared_core_constants.ACTION_UNLIKELY_INTENT_NAME
    )


def previous_action_was_listen(state: State) -> bool:
    """Check if action_listen is the previous executed action.

    Args:
        state: The state for which the check should be performed

    Return:
        boolean value indicating whether action_listen is previous action
    """
    return get_previous_action(state) == shared_core_constants.ACTION_LISTEN_NAME


########################################################################################
#                              State Manipulation
########################################################################################


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
        if last_state.get(USER, {}).get(shared_nlu_constants.ENTITIES):
            del last_state[USER][shared_nlu_constants.ENTITIES]
    else:
        # remove text features to only use intent
        if last_state.get(USER, {}).get(TEXT):
            del last_state[USER][TEXT]

    # "remove user text if intent"
    for state in states:
        # remove text features to only use intent
        if state.get(USER, {}).get(INTENT) and state.get(USER, {}).get(TEXT):
            del state[USER][TEXT]
