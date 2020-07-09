from typing import Dict, Text, List

import pytest
from aioresponses import aioresponses

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.actions.forms import FormAction, REQUESTED_SLOT
from rasa.core.channels import CollectingOutputChannel
from rasa.core.domain import Domain
from rasa.core.events import (
    Form,
    SlotSet,
    UserUttered,
    ActionExecuted,
    BotUttered,
    Restarted,
    Event,
)
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig


async def test_activate():
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=[])
    form_name = "my form"
    action = FormAction(form_name, None)
    slot_name = "num_people"
    domain = f"""
forms:
- {form_name}:
    {slot_name}:
    - type: from_entity
      entity: number
responses:
    utter_ask_num_people:
    - text: "How many people?"
"""
    domain = Domain.from_yaml(domain)

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.templates),
        tracker,
        domain,
    )
    assert events[:-1] == [Form(form_name), SlotSet(REQUESTED_SLOT, slot_name)]
    assert isinstance(events[-1], BotUttered)


async def test_activate_and_immediate_deactivate():
    slot_name = "num_people"
    slot_value = 5

    tracker = DialogueStateTracker.from_events(
        sender_id="bla",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                "haha",
                {"name": "greet"},
                entities=[{"entity": slot_name, "value": slot_value}],
            ),
        ],
    )
    form_name = "my form"
    action = FormAction(form_name, None)
    domain = f"""
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_entity
          entity: {slot_name}
    slots:
      {slot_name}:
        type: unfeaturized
    """
    domain = Domain.from_yaml(domain)
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.templates),
        tracker,
        domain,
    )
    assert events == [
        Form(form_name),
        SlotSet(slot_name, slot_value),
        SlotSet(REQUESTED_SLOT, None),
        Form(None),
    ]


async def test_set_slot_and_deactivate():
    form_name = "my form"
    slot_name = "num_people"
    slot_value = "dasdasdfasdf"
    events = [
        Form(form_name),
        SlotSet(REQUESTED_SLOT, slot_name),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(slot_value),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=events)

    domain = f"""
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_text
    slots:
      {slot_name}:
        type: unfeaturized
    """
    domain = Domain.from_yaml(domain)

    action = FormAction(form_name, None)
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.templates),
        tracker,
        domain,
    )
    assert events == [
        SlotSet(slot_name, slot_value),
        SlotSet(REQUESTED_SLOT, None),
        Form(None),
    ]


@pytest.mark.parametrize(
    "validate_return_events, expected_events",
    [
        # Validate function returns SlotSet events for every slot to fill
        (
            [
                {"event": "slot", "name": "num_people", "value": "so_clean"},
                {"event": "slot", "name": "num_tables", "value": 5},
            ],
            [
                SlotSet("num_people", "so_clean"),
                SlotSet("num_tables", 5),
                SlotSet(REQUESTED_SLOT, None),
                Form(None),
            ],
        ),
        # Validate function returns extra Slot Event
        (
            [
                {"event": "slot", "name": "num_people", "value": "so_clean"},
                {"event": "slot", "name": "some_other_slot", "value": 2},
            ],
            [
                SlotSet("num_people", "so_clean"),
                SlotSet("some_other_slot", 2),
                SlotSet("num_tables", 5),
                SlotSet(REQUESTED_SLOT, None),
                Form(None),
            ],
        ),
        # Validate function only validates one of the candidates
        (
            [{"event": "slot", "name": "num_people", "value": "so_clean"}],
            [
                SlotSet("num_people", "so_clean"),
                SlotSet("num_tables", 5),
                SlotSet(REQUESTED_SLOT, None),
                Form(None),
            ],
        ),
        # Validate function says slot is invalid
        (
            [{"event": "slot", "name": "num_people", "value": None}],
            [
                SlotSet("num_people", None),
                SlotSet("num_tables", 5),
                SlotSet(REQUESTED_SLOT, "num_people"),
            ],
        ),
        # Validate function decides to request a slot which is not part of the default
        # slot mapping
        (
            [{"event": "slot", "name": "requested_slot", "value": "is_outside"}],
            [
                SlotSet(REQUESTED_SLOT, "is_outside"),
                SlotSet("num_tables", 5),
                SlotSet("num_people", "hi"),
            ],
        ),
    ],
)
async def test_validate_slots(
    validate_return_events: List[Dict], expected_events: List[Event]
):
    form_name = "my form"
    slot_name = "num_people"
    slot_value = "hi"
    events = [
        Form(form_name),
        SlotSet(REQUESTED_SLOT, slot_name),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(slot_value, entities=[{"entity": "num_tables", "value": 5}]),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=events)

    domain = f"""
    slots:
      {slot_name}:
        type: unfeaturized
      num_tables:
        type: unfeaturized
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_text
        num_tables:
        - type: from_entity
          entity: num_tables
    actions:
    - validate_{form_name}
    """
    domain = Domain.from_yaml(domain)
    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(action_server_url, payload={"events": validate_return_events})

        action_server = EndpointConfig(action_server_url)
        action = FormAction(form_name, action_server)

        events = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.templates),
            tracker,
            domain,
        )
        assert events == expected_events


def test_name_of_utterance():
    form_name = "another_form"
    slot_name = "num_people"
    full_utterance_name = f"utter_ask_{form_name}_{slot_name}"

    domain = f"""
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_text
    responses:
        {full_utterance_name}:
        - text: "How many people?"
    """
    domain = Domain.from_yaml(domain)

    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses():
        action_server = EndpointConfig(action_server_url)
        action = FormAction(form_name, action_server)

        assert action._name_of_utterance(domain, slot_name) == full_utterance_name
        assert (
            action._name_of_utterance(domain, "another_slot")
            == "utter_ask_another_slot"
        )


def test_temporary_tracker():
    extra_slot = "some_slot"
    sender_id = "test"
    domain = Domain.from_yaml(
        f"""        slots:
          {extra_slot}:
            type: unfeaturized
        """
    )

    previous_events = [ActionExecuted(ACTION_LISTEN_NAME)]
    old_tracker = DialogueStateTracker.from_events(
        sender_id, previous_events, slots=domain.slots
    )
    new_events = [Restarted()]
    form_action = FormAction("some name", None)
    temp_tracker = form_action._temporary_tracker(old_tracker, new_events, domain)

    assert extra_slot in temp_tracker.slots.keys()
    assert list(temp_tracker.events) == [
        *previous_events,
        ActionExecuted(form_action.name()),
        *new_events,
    ]


def test_extract_requested_slot_default():
    """Test default extraction of a slot value from entity with the same name
    """
    form = FormAction("some form", None)

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", entities=[{"entity": "some_slot", "value": "some_value"}]
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = form.extract_requested_slot(tracker, Domain.empty())
    assert slot_values == {"some_slot": "some_value"}


@pytest.mark.parametrize(
    "slot_mapping, expected_value",
    [
        (
            {"type": "from_entity", "entity": "some_slot", "intent": "greet"},
            "some_value",
        ),
        (
            {"type": "from_intent", "intent": "greet", "value": "other_value"},
            "other_value",
        ),
        ({"type": "from_text"}, "bla"),
        ({"type": "from_text", "intent": "greet"}, "bla"),
        ({"type": "from_text", "not_intent": "other"}, "bla"),
    ],
)
def test_extract_requested_slot_when_mapping_applies(
    slot_mapping: Dict, expected_value: Text
):
    """Test extraction of a slot value from entity with the different name
        and certain intent
    """
    form_name = "some_form"
    entity_name = "some_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict({"forms": [{form_name: {entity_name: [slot_mapping]}}]})

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = form.extract_requested_slot(tracker, domain)
    # check that the value was extracted for correct intent
    assert slot_values == {"some_slot": expected_value}


@pytest.mark.parametrize(
    "slot_mapping",
    [
        {"type": "from_entity", "entity": "some_slot", "intent": "some_intent"},
        {"type": "from_intent", "intent": "some_intent", "value": "some_value"},
        {"type": "from_text", "intent": "other"},
        {"type": "from_text", "not_intent": "greet"},
    ],
)
def test_extract_requested_slot_mapping_does_not_apply(slot_mapping: Dict):
    """Test extraction of a slot value from entity with the different name
        and certain intent
    """
    form_name = "some_form"
    entity_name = "some_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict({"forms": [{form_name: {entity_name: [slot_mapping]}}]})

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = form.extract_requested_slot(tracker, domain)
    # check that the value was not extracted for incorrect intent
    assert slot_values == {}
