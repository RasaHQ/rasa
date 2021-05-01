from typing import List, Optional, Set
from rasa.shared.nlu.state_machine.state_machine_models import (
    Action,
    Intent,
    Slot,
    Utterance,
)
from rasa.shared.nlu.state_machine.conditions import Condition, IntentCondition

from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import CategoricalSlot, TextSlot, AnySlot
from rasa.shared.core.slots import Slot as RasaSlot
from rasa.shared.utils.io import dump_obj_as_yaml_to_string, write_text_file

from rasa.shared.nlu.training_data.formats import RasaYAMLReader

import rasa.shared.utils.validation
import rasa.shared.constants
import yaml


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
                states.update(next_state.all_states(checked_states=checked_states))

        return states.union({self})

    def get_intents_from_condition(self, condition: Condition) -> List[Intent]:
        if isinstance(condition, IntentCondition):
            return [condition.intent]
        else:
            return []

    def all_intents(self) -> Set[Intent]:
        intents: Set[Intent] = set()

        for state in self.all_states():
            response_conditions = [response.condition for response in state.responses]
            transition_conditions = [
                transition.condition for transition in state.transitions
            ]

            for condition in response_conditions + transition_conditions:
                intents_from_condition: List[Intent] = self.get_intents_from_condition(
                    condition
                )
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
            for prompt_actions in [slot.prompt_actions for slot in state.slots]:
                actions.update(prompt_actions)

            for response_actions in [response.actions for response in state.responses]:
                actions.update(response_actions)

            for transition_actions in [
                transition.transition_utterances for transition in state.transitions
            ]:
                actions.update(transition_actions)

            actions.update(state.slot_fill_utterances)

        return actions

    def all_slots(self) -> Set[Slot]:
        slots: Set[Slot] = set()

        for state in self.all_states():
            slots.update(state.slots)

        return slots

    # Write NLU
    def persist(self, states_filename: str, domain_filename: str, nlu_filename: str):
        domain, nlu_data = self.get_domain_nlu()

        # Persist domain
        rasa.shared.utils.validation.validate_yaml_schema(
            domain.as_yaml(), rasa.shared.constants.DOMAIN_SCHEMA_FILE
        )
        domain.persist(domain_filename)

        # Persist state
        states_yaml = yaml.dump([self])
        write_text_file(states_yaml, states_filename)

        # Persist NLU
        nlu_data["state_machine_files"] = states_filename
        nlu_data_yaml = dump_obj_as_yaml_to_string(
            nlu_data, should_preserve_key_order=True
        )
        RasaYAMLReader().validate(nlu_data_yaml)
        write_text_file(nlu_data_yaml, nlu_filename)

    def get_domain_nlu(self):
        all_entity_names = self.all_entities()
        all_intents: Set[Intent] = self.all_intents()
        all_utterances: Set[Utterance] = [
            action for action in self.all_actions() if isinstance(action, Utterance)
        ]
        all_actions: Set[Action] = self.all_actions()
        all_slots: Set[Slot] = self.all_slots()

        # Write domain
        domain = Domain(
            intents=[intent.name for intent in all_intents],
            entities=all_entity_names,  # List of entity names
            slots=[TextSlot(name=slot.name) for slot in all_slots],
            responses={
                utterance.name: [{"text": utterance.text}]
                for utterance in all_utterances
            },
            action_names=[action.name for action in all_actions],
            forms={},
            action_texts=[],
        )

        # Write NLU
        nlu_data = {
            "version": "2.0",
            "nlu": [intent.as_yaml() for intent in all_intents],
        }

        return domain, nlu_data