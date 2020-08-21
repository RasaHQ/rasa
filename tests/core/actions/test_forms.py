import asyncio
from typing import Dict, Text, List, Optional, Any
from unittest.mock import Mock, ANY

import pytest
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses

from rasa.core.actions import action
from rasa.core.actions.action import ACTION_LISTEN_NAME, ActionExecutionRejection
from rasa.core.actions.forms import FormAction, REQUESTED_SLOT
from rasa.core.channels import CollectingOutputChannel
from rasa.core.domain import Domain
from rasa.core.events import (
    ActiveLoop,
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
    assert events[:-1] == [ActiveLoop(form_name), SlotSet(REQUESTED_SLOT, slot_name)]
    assert isinstance(events[-1], BotUttered)


async def test_activate_with_prefilled_slot():
    slot_name = "num_people"
    slot_value = 5

    tracker = DialogueStateTracker.from_events(
        sender_id="bla", evts=[SlotSet(slot_name, slot_value)]
    )
    form_name = "my form"
    action = FormAction(form_name, None)

    next_slot_to_request = "next slot to request"
    domain = f"""
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_entity
          entity: {slot_name}
        {next_slot_to_request}:
        - type: from_text
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
        ActiveLoop(form_name),
        SlotSet(slot_name, slot_value),
        SlotSet(REQUESTED_SLOT, next_slot_to_request),
    ]


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
        ActiveLoop(form_name),
        SlotSet(slot_name, slot_value),
        SlotSet(REQUESTED_SLOT, None),
        ActiveLoop(None),
    ]


async def test_set_slot_and_deactivate():
    form_name = "my form"
    slot_name = "num_people"
    slot_value = "dasdasdfasdf"
    events = [
        ActiveLoop(form_name),
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
        ActiveLoop(None),
    ]


async def test_action_rejection():
    form_name = "my form"
    slot_to_fill = "some slot"
    tracker = DialogueStateTracker.from_events(
        sender_id="bla",
        evts=[
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, slot_to_fill),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("haha", {"name": "greet"}),
        ],
    )
    form_name = "my form"
    action = FormAction(form_name, None)
    domain = f"""
    forms:
    - {form_name}:
        {slot_to_fill}:
        - type: from_entity
          entity: some_entity
    slots:
      {slot_to_fill}:
        type: unfeaturized
    """
    domain = Domain.from_yaml(domain)

    with pytest.raises(ActionExecutionRejection):
        await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.templates),
            tracker,
            domain,
        )


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
                ActiveLoop(None),
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
                ActiveLoop(None),
            ],
        ),
        # Validate function only validates one of the candidates
        (
            [{"event": "slot", "name": "num_people", "value": "so_clean"}],
            [
                SlotSet("num_people", "so_clean"),
                SlotSet("num_tables", 5),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
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
        ActiveLoop(form_name),
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


async def test_validate_slots_on_activation_with_other_action_after_user_utterance():
    form_name = "my form"
    slot_name = "num_people"
    slot_value = "hi"
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(slot_value, entities=[{"entity": "num_tables", "value": 5}]),
        ActionExecuted("action_in_between"),
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
    - validate_{form_name}
    """
    domain = Domain.from_yaml(domain)
    action_server_url = "http:/my-action-server:5055/webhook"

    expected_slot_value = "✅"
    with aioresponses() as mocked:
        mocked.post(
            action_server_url,
            payload={
                "events": [
                    {"event": "slot", "name": slot_name, "value": expected_slot_value}
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
        ActiveLoop(form_name),
        SlotSet(slot_name, expected_slot_value),
        SlotSet(REQUESTED_SLOT, None),
        ActiveLoop(None),
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
    """Test default extraction of a slot value from entity with the same name."""
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
        {"type": "from_trigger_intent", "intent": "greet", "value": "value"},
    ],
)
def test_extract_requested_slot_mapping_does_not_apply(slot_mapping: Dict):
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
    "trigger_slot_mapping, expected_value",
    [
        ({"type": "from_trigger_intent", "intent": "greet", "value": "ten"}, "ten"),
        (
            {
                "type": "from_trigger_intent",
                "intent": ["bye", "greet"],
                "value": "tada",
            },
            "tada",
        ),
    ],
)
async def test_trigger_slot_mapping_applies(
    trigger_slot_mapping: Dict, expected_value: Text
):
    form_name = "some_form"
    entity_name = "some_slot"
    slot_filled_by_trigger_mapping = "other_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "forms": [
                {
                    form_name: {
                        entity_name: [
                            {
                                "type": "from_entity",
                                "entity": entity_name,
                                "intent": "some_intent",
                            }
                        ],
                        slot_filled_by_trigger_mapping: [trigger_slot_mapping],
                    }
                }
            ]
        }
    )

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

    slot_values = form.extract_other_slots(tracker, domain)
    assert slot_values == {slot_filled_by_trigger_mapping: expected_value}


@pytest.mark.parametrize(
    "trigger_slot_mapping",
    [
        ({"type": "from_trigger_intent", "intent": "bye", "value": "ten"}),
        ({"type": "from_trigger_intent", "not_intent": ["greet"], "value": "tada"}),
    ],
)
async def test_trigger_slot_mapping_does_not_apply(trigger_slot_mapping: Dict):
    form_name = "some_form"
    entity_name = "some_slot"
    slot_filled_by_trigger_mapping = "other_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "forms": [
                {
                    form_name: {
                        entity_name: [
                            {
                                "type": "from_entity",
                                "entity": entity_name,
                                "intent": "some_intent",
                            }
                        ],
                        slot_filled_by_trigger_mapping: [trigger_slot_mapping],
                    }
                }
            ]
        }
    )

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

    slot_values = form.extract_other_slots(tracker, domain)
    assert slot_values == {}


@pytest.mark.parametrize(
    "mapping_not_intent, mapping_intent, mapping_role, mapping_group, entities, intent, expected_slot_values",
    [
        (
            "some_intent",
            None,
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            None,
            "some_intent",
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {"some_slot": "some_value"},
        ),
        (
            "some_intent",
            None,
            None,
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_other_intent",
            {"some_slot": "some_value"},
        ),
        (
            None,
            None,
            "some_role",
            None,
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            None,
            None,
            "some_role",
            None,
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            {"some_slot": "some_value"},
        ),
        (
            None,
            None,
            None,
            "some_group",
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            None,
            None,
            None,
            "some_group",
            [{"entity": "some_entity", "value": "some_value", "group": "some_group"}],
            "some_intent",
            {"some_slot": "some_value"},
        ),
        (
            None,
            None,
            "some_role",
            "some_group",
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "group": "some_group",
                    "role": "some_role",
                }
            ],
            "some_intent",
            {"some_slot": "some_value"},
        ),
        (
            None,
            None,
            "some_role",
            "some_group",
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            {},
        ),
        (
            None,
            None,
            None,
            None,
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "group": "some_group",
                    "role": "some_role",
                }
            ],
            "some_intent",
            {"some_slot": "some_value"},
        ),
    ],
)
def test_extract_requested_slot_from_entity(
    mapping_not_intent: Optional[Text],
    mapping_intent: Optional[Text],
    mapping_role: Optional[Text],
    mapping_group: Optional[Text],
    entities: List[Dict[Text, Any]],
    intent: Text,
    expected_slot_values: Dict[Text, Text],
):
    """Test extraction of a slot value from entity with the different restrictions."""

    form_name = "some form"
    form = FormAction(form_name, None)

    mapping = form.from_entity(
        entity="some_entity",
        role=mapping_role,
        group=mapping_group,
        intent=mapping_intent,
        not_intent=mapping_not_intent,
    )
    domain = Domain.from_dict({"forms": [{form_name: {"some_slot": [mapping]}}]})

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", intent={"name": intent, "confidence": 1.0}, entities=entities
            ),
        ],
    )

    slot_values = form.extract_requested_slot(tracker, domain)
    assert slot_values == expected_slot_values


def test_invalid_slot_mapping():
    form_name = "my_form"
    form = FormAction(form_name, None)
    slot_name = "test"
    tracker = DialogueStateTracker.from_events(
        "sender", [SlotSet(REQUESTED_SLOT, slot_name)]
    )

    domain = Domain.from_dict(
        {"forms": [{form_name: {slot_name: [{"type": "invalid"}]}}]}
    )

    with pytest.raises(ValueError):
        form.extract_requested_slot(tracker, domain)


@pytest.mark.parametrize(
    "some_other_slot_mapping, some_slot_mapping, entities, intent, expected_slot_values",
    [
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "role": "some_role",
                }
            ],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "role": "some_other_role",
                }
            ],
            "some_intent",
            {},
        ),
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "role": "some_role",
                }
            ],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "group": "some_group",
                }
            ],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "group": "some_other_group",
                }
            ],
            "some_intent",
            {},
        ),
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "group": "some_group",
                }
            ],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [{"entity": "some_entity", "value": "some_value", "group": "some_group"}],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "group": "some_group",
                    "role": "some_role",
                }
            ],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [
                {
                    "entity": "some_entity",
                    "value": "some_value",
                    "role": "some_role",
                    "group": "some_group",
                }
            ],
            "some_intent",
            {"some_other_slot": "some_value"},
        ),
        (
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_other_entity",
                }
            ],
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            {},
        ),
        (
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_entity",
                    "role": "some_role",
                }
            ],
            [
                {
                    "type": "from_entity",
                    "intent": "some_intent",
                    "entity": "some_other_entity",
                }
            ],
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            {},
        ),
    ],
)
def test_extract_other_slots_with_entity(
    some_other_slot_mapping: List[Dict[Text, Any]],
    some_slot_mapping: List[Dict[Text, Any]],
    entities: List[Dict[Text, Any]],
    intent: Text,
    expected_slot_values: Dict[Text, Text],
):
    """Test extraction of other not requested slots values from entities."""

    form_name = "some_form"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "forms": [
                {
                    form_name: {
                        "some_other_slot": some_other_slot_mapping,
                        "some_slot": some_slot_mapping,
                    }
                }
            ]
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", intent={"name": intent, "confidence": 1.0}, entities=entities
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = form.extract_other_slots(tracker, domain)
    # check that the value was extracted for non requested slot
    assert slot_values == expected_slot_values


@pytest.mark.parametrize(
    "domain, expected_action",
    [
        ({}, "utter_ask_sun"),
        (
            {
                "actions": ["action_ask_my_form_sun", "action_ask_sun"],
                "responses": {"utter_ask_my_form_sun": [{"text": "ask"}]},
            },
            "action_ask_my_form_sun",
        ),
        (
            {
                "actions": ["action_ask_sun"],
                "responses": {"utter_ask_my_form_sun": [{"text": "ask"}]},
            },
            "utter_ask_my_form_sun",
        ),
        (
            {
                "actions": ["action_ask_sun"],
                "responses": {"utter_ask_sun": [{"text": "hi"}]},
            },
            "action_ask_sun",
        ),
        (
            {
                "actions": ["action_ask_my_form_sun"],
                "responses": {"utter_ask_my_form_sun": [{"text": "hi"}]},
            },
            "action_ask_my_form_sun",
        ),
    ],
)
async def test_ask_for_slot(
    domain: Dict, expected_action: Text, monkeypatch: MonkeyPatch
):
    slot_name = "sun"

    action_from_name = Mock(return_value=action.ActionListen())
    monkeypatch.setattr(action, action.action_from_name.__name__, action_from_name)

    form = FormAction("my_form", None)
    await form._ask_for_slot(
        Domain.from_dict(domain),
        None,
        None,
        slot_name,
        DialogueStateTracker.from_events("dasd", []),
    )

    action_from_name.assert_called_once_with(expected_action, None, ANY)
