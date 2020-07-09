from typing import Dict, Text, Optional, List, Any

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


async def test_validate_slots():
    form_name = "my form"
    slot_name = "num_people"
    slot_value = "dasdasdfasdf"
    validated_slot_value = "so clean"
    events = [
        Form(form_name),
        SlotSet(REQUESTED_SLOT, slot_name),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(slot_value),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=events)

    domain = f"""
    slots:
      {slot_name}:
        type: unfeaturized
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_text
    actions:
    - validate_{slot_name}
    """
    domain = Domain.from_yaml(domain)
    action_server_url = "http:/my-action-server:5055/webhook"

    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": slot_name, "value": validated_slot_value}
                ]
            },
        )

        action_server = EndpointConfig(action_server_url)
        action = FormAction(form_name, action_server)

        events = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.templates),
            tracker,
            domain,
        )
        assert events == [
            SlotSet(slot_name, validated_slot_value),
            SlotSet(REQUESTED_SLOT, None),
            Form(None),
        ]


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
        f"""
        slots:
          {extra_slot}:
            type: unfeaturized
        """
    )

    old_tracker = DialogueStateTracker.from_events(
        sender_id, [ActionExecuted(ACTION_LISTEN_NAME)], slots=domain.slots
    )
    new_events = [Restarted()]
    temp_tracker = FormAction._temporary_tracker(old_tracker, new_events, domain)

    assert extra_slot in temp_tracker.slots.keys()
    assert len(temp_tracker.events) == 2


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


@pytest.mark.parametrize(
    "mapping_entity, mapping_role, mapping_group, entities, intent, expected_slot_values",
    [
        (
            "entity_type",
            "some_role",
            None,
            [
                {
                    "entity": "entity_type",
                    "value": "some_value",
                    "role": "some_other_role",
                }
            ],
            "some_intent",
            {},
        ),
        (
            "entity_type",
            "some_role",
            None,
            [{"entity": "entity_type", "value": "some_value", "role": "some_role"}],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            "entity_type",
            None,
            "some_group",
            [
                {
                    "entity": "entity_type",
                    "value": "some_value",
                    "group": "some_other_group",
                }
            ],
            "some_intent",
            {},
        ),
        (
            "entity_type",
            None,
            "some_group",
            [{"entity": "entity_type", "value": "some_value", "group": "some_group"}],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            "entity_type",
            "some_role",
            "some_group",
            [
                {
                    "entity": "entity_type",
                    "value": "some_value",
                    "role": "some_role",
                    "group": "some_group",
                }
            ],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            "entity_type",
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            "some_entity",
            None,
            None,
            [{"entity": "entity_type", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            "entity_type",
            None,
            None,
            [{"entity": "entity_type", "value": "some_value"}],
            "some_intent",
            {},
        ),
    ],
)
def test_extract_other_slots_with_entity(
    mapping_entity: Text,
    mapping_role: Optional[Text],
    mapping_group: Optional[Text],
    entities: List[Dict[Text, Any]],
    intent: Text,
    expected_slot_values: Dict[Text, Text],
):
    """Test extraction of other not requested slots values from entities."""

    form_name = "some_form"
    slot_mapping = {
        "type": "from_entity",
        "entity": mapping_entity,
        "role": mapping_role,
        "group": mapping_group,
    }
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {"forms": [{form_name: {"some_other_slot": [slot_mapping]}}]}
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, ["some_slot", "some_other_slot"]),
            UserUttered(
                "bla", intent={"name": intent, "confidence": 1.0}, entities=entities
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = form.extract_other_slots(tracker, domain)
    # check that the value was extracted for non requested slot
    assert slot_values == expected_slot_values
