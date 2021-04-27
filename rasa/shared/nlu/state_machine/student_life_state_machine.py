from typing import Any, Dict, List

from rasa.shared.nlu.state_machine.state_machine_models import (
    Intent,
    Utterance,
    Slot,
)

from rasa.shared.nlu.state_machine.state_machine_state import (
    Action,
    Response,
    StateMachineState,
    Transition,
)

from rasa.shared.nlu.state_machine.conditions import (
    AndCondition,
    IntentCondition,
    OnEntryCondition,
    SlotEqualsCondition,
)

from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import CategoricalSlot, TextSlot, AnySlot
from rasa.shared.core.slots import Slot as RasaSlot
from rasa.shared.utils.io import dump_obj_as_yaml_to_string, write_text_file

# class SpaceEntity(Enum, Entity):
#     person = "PERSON"
#     geopolitical_entity = "GPE"
#     location = "LOC"

#     def name(self) -> str:
#         return self.value


wantToLeaveIntent = Intent(name="want_to_leave", examples=["I want to leave"])
whereAreYouFromIntent = Intent(
    name="where_are_you_from", examples=["Where are you from?"]
)
wheresTheWashroomIntent = Intent(
    name="wheres_the_washroom",
    examples=[
        "Where's the washroom?",
        "Where is the restroom?",
        "What is the location of the washroom?",
        "I need to find the toilet",
    ],
)
howAreYouDoingIntent = Intent(name="how_are_you_doing", examples=["How are you doing?"])

slotName = Slot(
    name="name",
    entities=["PERSON"],
    prompt_actions=[
        Utterance(
            text="Can I get your name?",
            name="utter_can_i_get_your_name",
        ),
        Utterance(
            text="What about your name?",
            name="utter_what_about_your_name",
        ),
    ],
)

slotHometown = Slot(
    name="hometown",
    entities=["GPE", "LOC"],
    prompt_actions=[
        Utterance(text="What is your hometown?", name="utter_what_is_your_hometown")
    ],
)

generalResponses: List[Response] = [
    Response(
        condition=IntentCondition(whereAreYouFromIntent),
        actions=[Utterance(text="I'm from Canada", name="utter_where_from_response")],
    ),
    Response(
        condition=IntentCondition(wheresTheWashroomIntent),
        actions=[
            Utterance(
                text="It's in the cafeteria",
                name="utter_washroom_response",
            )
        ],
    ),
    Response(
        condition=AndCondition(
            [
                IntentCondition(howAreYouDoingIntent),
                SlotEqualsCondition(slotName, "Alice"),
                SlotEqualsCondition(slotHometown, "Austin"),
            ]
        ),
        actions=[
            Utterance(
                text="I'm doing great",
                name="utter_how_are_you_response",
            )
        ],
    ),
]

student_life_state_machine = StateMachineState(
    name="student_form",
    slots=[slotName, slotHometown],
    slot_fill_utterances=[
        Utterance(text="Nice to meet you {name}", name="utter_greeting_response"),
        Utterance(
            text="I'd love to visit {hometown} someday",
            name="utter_hometown_slot_filled_response",
        ),
    ],
    transitions=[
        Transition(
            name="exit_form",
            condition=IntentCondition(wantToLeaveIntent),
            transition_utterances=[
                Utterance(
                    text="Sure, let's go back to what we were talking about.",
                    name="utter_leave_response",
                )
            ],
            destination_state=None,
        )
    ],
    responses=[
        Response(
            condition=OnEntryCondition(),
            actions=[
                Utterance(
                    text="I'll need some more info from you",
                    name="utter_need_more_info",
                )
            ],
        )
    ]
    + generalResponses,
)


def convert_to_rasa_slot(slot: Slot) -> RasaSlot:
    return TextSlot(name=slot.name)


def convert_intent_to_nlu(intent: Intent) -> Dict[str, Any]:
    return {"intent": intent.name, "examples": intent.examples}


# Write NLU
def writeNLU(state: StateMachineState, domain_filename: str, nlu_filename: str):
    all_entity_names = state.all_entities()
    all_intents: Set[Intent] = state.all_intents()
    all_utterances: Set[Utterance] = [
        action for action in state.all_actions() if isinstance(action, Utterance)
    ]
    all_actions: Set[Action] = state.all_actions()
    all_slots: Set[Slot] = state.all_slots()

    # Write domain
    domain = Domain(
        intents=[intent.name for intent in all_intents],
        entities=all_entity_names,  # List of entity names
        slots=[convert_to_rasa_slot(slot) for slot in all_slots],
        responses={
            utterance.name: [{"text": utterance.text}] for utterance in all_utterances
        },
        action_names=[action.name for action in all_actions],
        forms={},
        action_texts=[],
    )
    domain.persist(domain_filename)

    # Write NLU
    nlu_data = {
        "version": "2.0",
        "nlu": [convert_intent_to_nlu(intent) for intent in all_intents],
    }

    nlu_data_yaml = dump_obj_as_yaml_to_string(nlu_data, should_preserve_key_order=True)

    write_text_file(nlu_data_yaml, nlu_filename)


writeNLU(
    state=student_life_state_machine,
    domain_filename="state_machine_domain.yaml",
    nlu_filename="state_machine_nlu.yaml",
)
