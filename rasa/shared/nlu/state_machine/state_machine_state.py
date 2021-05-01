from typing import List, Optional, Set
from rasa.shared.nlu.state_machine.state_machine_models import (
    Action,
    Intent,
    Slot,
    Utterance,
)
from rasa.shared.nlu.state_machine.conditions import Condition, IntentCondition


class Response:
    condition: Condition
    repeatable: bool
    actions: List[str]

    def __init__(
        self,
        condition: Condition,
        actions: List[Action],
        repeatable: bool = True,
    ):
        self.condition = condition
        self.repeatable = repeatable
        self.actions = actions


class Transition:
    name: str
    condition: Condition
    transition_utterances: List[Utterance]
    destination_state: Optional[str]

    def __init__(
        self,
        name: str,
        condition: Condition,
        transition_utterances: List[Utterance],
        destination_state: Optional["StateMachineState"],
    ):
        self.name = name
        self.condition = condition
        self.transition_utterances = transition_utterances
        self.destination_state = destination_state


class StateMachineState:
    name: str
    slots: List[Slot]
    slot_fill_utterances: List[str]
    transitions: List[Transition]
    responses: List[Response]

    def __init__(
        self,
        name: str,
        slots: List[Slot],
        slot_fill_utterances: List[Utterance],
        transitions: List[Transition],
        responses: List[Response],
    ):
        self.name = name
        self.slots = slots
        self.slot_fill_utterances = slot_fill_utterances
        self.transitions = transitions
        self.responses = responses

    def all_states(
        self,
        checked_states: Set["StateMachineState"] = {},
    ) -> Set["StateMachineState"]:
        states: Set["StateMachineState"] = set()
        for transition in self.transitions:
            next_state = transition.destination_state
            if next_state and next_state not in checked_states:
                states.update(
                    next_state.all_states(checked_states=checked_states)
                )

        return states.union({self})

    def get_intents_from_condition(self, condition: Condition) -> List[Intent]:
        if isinstance(condition, IntentCondition):
            return [condition.intent]
        else:
            return []

    def all_intents(self) -> Set[Intent]:
        intents: Set[Intent] = set()

        for state in self.all_states():
            response_conditions = [
                response.condition for response in state.responses
            ]
            transition_conditions = [
                transition.condition for transition in state.transitions
            ]

            for condition in response_conditions + transition_conditions:
                intents_from_condition: List[
                    Intent
                ] = self.get_intents_from_condition(condition)
                intents.update(intents_from_condition)

        return intents

    def all_entities(self) -> Set[str]:
        entities: Set[str] = set()

        for state in self.all_states():
            for slot in state.slots:
                entities.update(slot.entities)

        return entities

    def all_actions(self) -> Set[Action]:
        actions: Set[Action] = set()

        for state in self.all_states():
            for prompt_actions in [
                slot.prompt_actions for slot in state.slots
            ]:
                actions.update(prompt_actions)

            for response_actions in [
                response.actions for response in state.responses
            ]:
                actions.update(response_actions)

            for transition_actions in [
                transition.transition_utterances
                for transition in state.transitions
            ]:
                actions.update(transition_actions)

            actions.update(state.slot_fill_utterances)

        return actions

    def all_slots(self) -> Set[Slot]:
        slots: Set[Slot] = set()

        for state in self.all_states():
            slots.update(state.slots)

        return slots
