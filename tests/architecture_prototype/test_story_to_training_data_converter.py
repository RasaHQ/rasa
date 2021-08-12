from rasa.shared.core.constants import DEFAULT_ACTION_NAMES
from rasa.shared.core.slots import Slot
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import INTENT_NAME_KEY
from rasa.architecture_prototype.graph_components import (
    E2ELookupTable,
    StoryToTrainingDataConverter,
)
from rasa.shared.core.events import UserUttered, ActionExecuted


def test_convert_events():
    events = [
        UserUttered(intent={INTENT_NAME_KEY: "greet"}),
        ActionExecuted(action_name="utter_greet"),
        ActionExecuted(action_name="utter_greet"),
    ]
    lookup_table = E2ELookupTable(handle_collisions=False)
    StoryToTrainingDataConverter.add_sub_states_from_events(events, lookup_table)
    # should create: one intent-only user substate and one prev_action substate
    assert len(lookup_table) == 2

    events = [
        UserUttered(text="text", intent={INTENT_NAME_KEY: "greet"}),
        ActionExecuted(action_name="utter_greet"),
    ]
    lookup_table = E2ELookupTable(handle_collisions=False)
    StoryToTrainingDataConverter.add_sub_states_from_events(events, lookup_table)
    # should create: one intent-only user substate, one text-only user substate one
    # prev_action substate
    assert len(lookup_table) == 3


def test_convert_domain():
    action_names = ["a", "b"]
    # action texts, response keys, forms, and action_names must be unique or the
    # domain will complain about it
    action_texts = ["a2", "b2"]
    responses = {"a3": "a2", "b3": "b2"}
    forms = ["a4"]
    # however, intent names can be anything
    intents = ["a", "b"]
    domain = Domain(
        intents=intents,
        action_names=action_names,
        action_texts=action_texts,
        responses=responses,
        entities=["e_a", "e_b", "e_c"],
        slots=[Slot(name="s")],
        forms=forms,
    )
    lookup_table = E2ELookupTable(handle_collisions=True)
    StoryToTrainingDataConverter.add_sub_states_from_domain(domain, lookup_table)
    # Note that we cannot just sum the above because the `domain` will e.g. combine the
    # `action_texts` with the `responses` to obtain the `domain.action_texts`
    assert len(lookup_table) == (
        len(domain.intent_properties)
        + len(domain.user_actions)
        + len(domain.action_texts)
        + len(DEFAULT_ACTION_NAMES)
    )
