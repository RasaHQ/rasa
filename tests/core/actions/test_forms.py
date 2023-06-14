import logging
import textwrap
from typing import Dict, Text, List, Any, Union
from unittest.mock import Mock

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses

from rasa.core.agent import Agent
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.actions import action
from rasa.core.actions.action import ActionExecutionRejection, ActionExtractSlots
from rasa.shared.constants import (
    LATEST_TRAINING_DATA_FORMAT_VERSION,
    REQUIRED_SLOTS_KEY,
    IGNORED_INTENTS,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME, REQUESTED_SLOT
from rasa.core.actions.forms import FormAction
from rasa.core.channels import CollectingOutputChannel
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActiveLoop,
    SlotSet,
    UserUttered,
    ActionExecuted,
    BotUttered,
    Restarted,
    Event,
    ActionExecutionRejected,
    DefinePrevUserUtteredFeaturization,
)
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig

ACTION_SERVER_URL = "http://my-action-server:5055/webhook"


async def test_activate():
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=[])
    form_name = "my form"
    action = FormAction(form_name, None)
    slot_name = "num_people"
    domain = textwrap.dedent(
        f"""
    slots:
      {slot_name}:
        type: float
        mappings:
        - type: from_entity
          entity: number
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
        - {slot_name}
    responses:
      utter_ask_num_people:
      - text: "How many people?"
      """
    )
    domain = Domain.from_yaml(domain)

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert isinstance(events[-1], BotUttered)
    assert events[:-1] == [ActiveLoop(form_name), SlotSet(REQUESTED_SLOT, slot_name)]


async def test_activate_with_custom_slot_mapping():
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=[])
    form_name = "my_form"
    action_server = EndpointConfig(ACTION_SERVER_URL)
    action = FormAction(form_name, action_server)
    domain_required_slot_name = "num_people"
    slot_set_by_remote_custom_extraction_method = "some_slot"
    slot_value_set_by_remote_custom_extraction_method = "anything"
    domain = textwrap.dedent(
        f"""
    slots:
      {domain_required_slot_name}:
        type: float
        mappings:
        - type: from_entity
          entity: number
      {slot_set_by_remote_custom_extraction_method}:
          type: any
          mappings:
          - type: custom
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
        - {domain_required_slot_name}
    responses:
      utter_ask_num_people:
      - text: "How many people?"
    actions:
      - validate_{form_name}
      """
    )
    domain = Domain.from_yaml(domain)

    form_validation_events = [
        {
            "event": "slot",
            "timestamp": None,
            "name": slot_set_by_remote_custom_extraction_method,
            "value": slot_value_set_by_remote_custom_extraction_method,
        }
    ]
    with aioresponses() as mocked:
        mocked.post(
            ACTION_SERVER_URL,
            payload={"events": form_validation_events},
        )
        events = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
    assert events[:-1] == [
        ActiveLoop(form_name),
        SlotSet(
            slot_set_by_remote_custom_extraction_method,
            slot_value_set_by_remote_custom_extraction_method,
        ),
        SlotSet(REQUESTED_SLOT, domain_required_slot_name),
    ]
    assert isinstance(events[-1], BotUttered)


async def test_activate_with_mapping_conditions_slot():
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=[])
    form_name = "my form"
    action = FormAction(form_name, None)
    slot_name = "num_people"
    domain = textwrap.dedent(
        f"""
    slots:
      {slot_name}:
        type: float
        mappings:
        - type: from_entity
          entity: number
          conditions:
          - active_loop: {form_name}
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
        - {slot_name}
    responses:
      utter_ask_num_people:
      - text: "How many people?"
      """
    )
    domain = Domain.from_yaml(domain)

    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
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
    form_name = "my_form"
    action = FormAction(form_name, None)

    next_slot_to_request = "next slot to request"
    domain = f"""
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
            - {next_slot_to_request}
    slots:
      {slot_name}:
        type: any
        mappings:
        - type: from_entity
          entity: {slot_name}
      {next_slot_to_request}:
        type: text
        mappings:
        - type: from_text
    """
    domain = Domain.from_yaml(domain)
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, next_slot_to_request),
    ]


async def test_activate_with_prefilled_slot_with_mapping_conditions():
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
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
            - {next_slot_to_request}
    slots:
      {slot_name}:
        type: any
        mappings:
        - type: from_entity
          entity: {slot_name}
          conditions:
          - active_loop: {form_name}
      {next_slot_to_request}:
        type: text
        mappings:
        - type: from_text
          conditions:
          - active_loop: {form_name}
    """
    domain = Domain.from_yaml(domain)
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, next_slot_to_request),
    ]


async def test_switch_forms_with_same_slot(default_agent: Agent):
    """Tests switching of forms, where the first slot is the same in both forms.

    Tests the fix for issue 7710"""

    # Define two forms in the domain, with same first slot
    slot_a = "my_slot_a"

    form_1 = "my_form_1"
    utter_ask_form_1 = f"Please provide the value for {slot_a} of form 1"

    form_2 = "my_form_2"
    utter_ask_form_2 = f"Please provide the value for {slot_a} of form 2"

    domain = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- intent: order_status
  examples: |
    - check status of my order
    - when are my shoes coming in
- intent: return
  examples: |
    - start a return
    - I don't want my shoes anymore
slots:
 {slot_a}:
   type: float
   mappings:
   - type: from_entity
     entity: number
forms:
  {form_1}:
    {REQUIRED_SLOTS_KEY}:
        - {slot_a}
  {form_2}:
    {REQUIRED_SLOTS_KEY}:
        - {slot_a}
responses:
    utter_ask_{form_1}_{slot_a}:
    - text: {utter_ask_form_1}
    utter_ask_{form_2}_{slot_a}:
    - text: {utter_ask_form_2}
"""

    domain = Domain.from_yaml(domain)

    # Driving it like rasa/core/processor
    processor = default_agent.processor
    processor.domain = domain

    # activate the first form
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("order status", {"name": "form_1", "confidence": 1.0}),
            DefinePrevUserUtteredFeaturization(False),
        ],
    )
    # rasa/core/processor.predict_next_action
    prediction = PolicyPrediction([], "some_policy")
    action_1 = FormAction(form_1, None)

    await processor._run_action(
        action_1,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        prediction,
    )

    events_expected = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("order status", {"name": "form_1", "confidence": 1.0}),
        DefinePrevUserUtteredFeaturization(False),
        ActionExecuted(form_1),
        ActiveLoop(form_1),
        SlotSet(REQUESTED_SLOT, slot_a),
        BotUttered(
            text=utter_ask_form_1,
            metadata={"utter_action": f"utter_ask_{form_1}_{slot_a}"},
        ),
    ]
    assert tracker.applied_events() == events_expected

    next_events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("return my shoes", {"name": "form_2", "confidence": 1.0}),
        DefinePrevUserUtteredFeaturization(False),
    ]
    tracker.update_with_events(next_events, domain)
    events_expected.extend(next_events)

    # form_1 is still active, and bot will first validate if the user utterance
    #  provides valid data for the requested slot, which is rejected
    await processor._run_action(
        action_1,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        prediction,
    )
    events_expected.extend([ActionExecutionRejected(action_name=form_1)])
    assert tracker.applied_events() == events_expected

    # Next, bot predicts form_2
    action_2 = FormAction(form_2, None)
    await processor._run_action(
        action_2,
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        prediction,
    )
    events_expected.extend(
        [
            ActionExecuted(form_2),
            ActiveLoop(form_2),
            SlotSet(REQUESTED_SLOT, slot_a),
            BotUttered(
                text=utter_ask_form_2,
                metadata={"utter_action": f"utter_ask_{form_2}_{slot_a}"},
            ),
        ]
    )
    assert tracker.applied_events() == events_expected


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
            SlotSet(slot_name, slot_value),
        ],
    )
    form_name = "my form"
    action = FormAction(form_name, None)
    domain = f"""
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
    slots:
      {slot_name}:
        type: any
        mappings:
        - type: from_entity
          entity: {slot_name}
    """
    domain = Domain.from_yaml(domain)
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [
        ActiveLoop(form_name),
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
        SlotSet(slot_name, slot_value),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=events)

    domain = f"""
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
    slots:
      {slot_name}:
        type: text
        influence_conversation: false
        mappings:
        - type: from_text
    """
    domain = Domain.from_yaml(domain)

    form_action = FormAction(form_name, None)
    events = await form_action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == [SlotSet(REQUESTED_SLOT, None), ActiveLoop(None)]


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
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_to_fill}
    slots:
      {slot_to_fill}:
        type: any
        mappings:
        - type: from_entity
          entity: some_entity
    """
    domain = Domain.from_yaml(domain)

    with pytest.raises(ActionExecutionRejection):
        await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
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
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        # Validate function only validates one of the candidates
        (
            [{"event": "slot", "name": "num_people", "value": "so_clean"}],
            [
                SlotSet("num_people", "so_clean"),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        # Validate function says slot is invalid
        (
            [{"event": "slot", "name": "num_people", "value": None}],
            [SlotSet("num_people", None), SlotSet(REQUESTED_SLOT, "num_people")],
        ),
        # Validate function decides to request a slot which is not part of the default
        # slot mapping
        (
            [{"event": "slot", "name": "requested_slot", "value": "is_outside"}],
            [SlotSet(REQUESTED_SLOT, "is_outside")],
        ),
        # Validate function decides that no more slots should be requested
        (
            [
                {"event": "slot", "name": "num_people", "value": None},
                {"event": "slot", "name": REQUESTED_SLOT, "value": None},
            ],
            [
                SlotSet("num_people", None),
                SlotSet(REQUESTED_SLOT, None),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        # Validate function deactivates loop
        (
            [
                {"event": "slot", "name": "num_people", "value": None},
                {"event": "active_loop", "name": None},
            ],
            [
                SlotSet("num_people", None),
                ActiveLoop(None),
                SlotSet(REQUESTED_SLOT, None),
                ActiveLoop(None),
            ],
        ),
        # User rejected manually
        (
            [{"event": "action_execution_rejected", "name": "my form"}],
            [ActionExecutionRejected("my form")],
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
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

    entities:
    - num_tables
    - some_entity
    - some_other_entity
    - some_slot

    slots:
      {slot_name}:
        type: any
        mappings:
        - type: from_text
      num_tables:
        type: any
        mappings:
        - type: from_entity
          entity: num_tables

    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
            - num_tables

    actions:
    - validate_{form_name}
    """
    domain = Domain.from_yaml(textwrap.dedent(domain))
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert slot_events == [SlotSet(slot_name, slot_value), SlotSet("num_tables", 5)]
    tracker.update_with_events(slot_events, domain)

    with aioresponses() as mocked:
        mocked.post(ACTION_SERVER_URL, payload={"events": validate_return_events})

        action_server = EndpointConfig(ACTION_SERVER_URL)
        action = FormAction(form_name, action_server)

        events = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert events == expected_events


async def test_request_correct_slots_after_unhappy_path_with_custom_required_slots():
    form_name = "some_form"
    slot_name_1 = "slot_1"
    slot_name_2 = "slot_2"

    domain = f"""
        slots:
          {slot_name_1}:
            type: any
            mappings:
            - type: from_intent
              intent: some_intent
              value: some_value
          {slot_name_2}:
            type: any
            mappings:
            - type: from_intent
              intent: some_intent
              value: some_value
        forms:
          {form_name}:
            {REQUIRED_SLOTS_KEY}:
                - {slot_name_1}
                - {slot_name_2}
        actions:
        - validate_{form_name}
        """
    domain = Domain.from_yaml(domain)

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop(form_name),
            SlotSet(REQUESTED_SLOT, "slot_2"),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("hello", intent={"name": "greet", "confidence": 1.0}),
            ActionExecutionRejected(form_name),
            ActionExecuted("utter_greet"),
        ],
    )

    # Custom form validation action changes the order of the requested slots
    validate_return_events = [
        {"event": "slot", "name": REQUESTED_SLOT, "value": slot_name_2}
    ]

    # The form should ask the same slot again when coming back after unhappy path
    expected_events = [SlotSet(REQUESTED_SLOT, slot_name_2)]

    with aioresponses() as mocked:
        mocked.post(ACTION_SERVER_URL, payload={"events": validate_return_events})

        action_server = EndpointConfig(ACTION_SERVER_URL)
        action = FormAction(form_name, action_server)

        events = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert events == expected_events


@pytest.mark.parametrize(
    "custom_events",
    [
        # Custom action returned no events
        [],
        # Custom action returned events but no `SlotSet` events
        [BotUttered("some text").as_dict()],
        # Custom action returned only `SlotSet` event for `required_slot`
        [SlotSet(REQUESTED_SLOT, "some value").as_dict()],
    ],
)
async def test_no_slots_extracted_with_custom_slot_mappings(custom_events: List[Event]):
    form_name = "my form"
    events = [
        ActiveLoop(form_name),
        SlotSet(REQUESTED_SLOT, "num_tables"),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered("off topic"),
    ]
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=events)

    domain = f"""
    slots:
      num_tables:
        type: any
        mappings:
        - type: from_entity
          entity: num_tables
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - num_tables
    actions:
    - validate_{form_name}
    """
    domain = Domain.from_yaml(domain)

    with aioresponses() as mocked:
        mocked.post(ACTION_SERVER_URL, payload={"events": custom_events})

        action_server = EndpointConfig(ACTION_SERVER_URL)
        action = FormAction(form_name, action_server)

        with pytest.raises(ActionExecutionRejection):
            await action.run(
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
                tracker,
                domain,
            )


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
        type: any
        mappings:
        - type: from_text
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
    actions:
    - validate_{form_name}
    """
    domain = Domain.from_yaml(domain)
    expected_slot_value = "✅"
    with aioresponses() as mocked:
        mocked.post(
            ACTION_SERVER_URL,
            payload={
                "events": [
                    {"event": "slot", "name": slot_name, "value": expected_slot_value}
                ]
            },
        )

        action_server = EndpointConfig(ACTION_SERVER_URL)
        action_extract_slots = ActionExtractSlots(action_endpoint=None)
        slot_events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        tracker.update_with_events(slot_events, domain)

        form_action = FormAction(form_name, action_server)

        mocked.post(
            ACTION_SERVER_URL,
            payload={"events": []},
        )
        events = await form_action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )

    assert events == [
        ActiveLoop(form_name),
        SlotSet(slot_name, expected_slot_value),
        SlotSet(REQUESTED_SLOT, None),
        ActiveLoop(None),
    ]


@pytest.mark.parametrize(
    "utterance_name", ["utter_ask_my_form_num_people", "utter_ask_num_people"]
)
def test_name_of_utterance(utterance_name: Text):
    form_name = "my_form"
    slot_name = "num_people"

    domain = f"""
    slots:
      {slot_name}:
        type: any
        mappings:
        - type: from_text
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
    responses:
        {utterance_name}:
        - text: "How many people?"
    """
    domain = Domain.from_yaml(domain)

    action = FormAction(form_name, None)

    assert action._name_of_utterance(domain, slot_name) == utterance_name


def test_temporary_tracker():
    extra_slot = "some_slot"
    sender_id = "test"
    domain = Domain.from_yaml(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
        slots:
          {extra_slot}:
            type: any
            mappings:
            - type: from_text
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
        SlotSet(REQUESTED_SLOT),
        ActionExecuted(form_action.name()),
        *new_events,
    ]


async def test_extract_requested_slot_default():
    """Test default extraction of a slot value from entity with the same name."""
    form_name = "some_form"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "slots": {
                "some_slot": {
                    "type": "text",
                    "mappings": [
                        {
                            "type": "from_entity",
                            "entity": "some_slot",
                            "value": "some_value",
                        }
                    ],
                }
            },
            "forms": {form_name: {REQUIRED_SLOTS_KEY: ["some_slot"]}},
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop("some form"),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", entities=[{"entity": "some_slot", "value": "some_value"}]
            ),
            SlotSet("some_slot", "some_value"),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    slot_values = await form.validate(
        tracker,
        domain,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
    )
    assert slot_values == []


@pytest.mark.parametrize(
    "some_other_slot_mapping, some_slot_mapping, entities, "
    "intent, expected_slot_events",
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
            [],
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
            [SlotSet("some_other_slot", "some_value")],
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
            [],
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
            [SlotSet("some_other_slot", "some_value")],
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
            [SlotSet("some_other_slot", "some_value")],
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
            # other slot should be extracted because slot mapping is unique
            [SlotSet("some_other_slot", "some_value")],
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
            # other slot should be extracted because slot mapping is unique
            [SlotSet("some_other_slot", "some_value")],
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
            [{"entity": "some_entity", "value": "some_value", "role": "some_role"}],
            "some_intent",
            # other slot should not be extracted
            # because even though slot mapping is unique it doesn't contain the role
            [],
        ),
        (
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [{"type": "from_entity", "intent": "some_intent", "entity": "some_entity"}],
            [{"entity": "some_entity", "value": "some_value"}],
            "some_intent",
            # other slot should not be extracted because slot mapping is not unique
            [SlotSet("some_slot", "some_value")],
        ),
    ],
)
async def test_extract_other_slots_with_entity(
    some_other_slot_mapping: List[Dict[Text, Any]],
    some_slot_mapping: List[Dict[Text, Any]],
    entities: List[Dict[Text, Any]],
    intent: Text,
    expected_slot_events: List[SlotSet],
):
    """Test extraction of other not requested slots values from entities."""
    form_name = "some_form"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "slots": {
                "some_other_slot": {"type": "any", "mappings": some_other_slot_mapping},
                "some_slot": {"type": "any", "mappings": some_slot_mapping},
            },
            "forms": {
                form_name: {REQUIRED_SLOTS_KEY: ["some_other_slot", "some_slot"]}
            },
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "bla", intent={"name": intent, "confidence": 1.0}, entities=entities
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(slot_events, domain)

    if slot_events:
        slot_values = await form.validate(
            tracker,
            domain,
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
        )
        assert slot_values == expected_slot_events
    else:
        with pytest.raises(ActionExecutionRejection):
            await form.validate(
                tracker,
                domain,
                CollectingOutputChannel(),
                TemplatedNaturalLanguageGenerator(domain.responses),
            )


@pytest.mark.parametrize(
    "domain_dict, expected_action",
    [
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
    domain_dict: Dict,
    expected_action: Text,
    monkeypatch: MonkeyPatch,
    default_nlg: TemplatedNaturalLanguageGenerator,
):
    slot_name = "sun"

    action_from_name = Mock(return_value=action.ActionListen())
    endpoint_config = Mock()
    monkeypatch.setattr(
        action, action.action_for_name_or_text.__name__, action_from_name
    )

    form = FormAction("my_form", endpoint_config)
    domain = Domain.from_dict(domain_dict)
    await form._ask_for_slot(
        domain,
        default_nlg,
        CollectingOutputChannel(),
        slot_name,
        DialogueStateTracker.from_events("dasd", []),
    )

    action_from_name.assert_called_once_with(expected_action, domain, endpoint_config)


async def test_ask_for_slot_if_not_utter_ask(
    monkeypatch: MonkeyPatch, default_nlg: TemplatedNaturalLanguageGenerator
):
    action_from_name = Mock(return_value=action.ActionListen())
    endpoint_config = Mock()
    monkeypatch.setattr(
        action, action.action_for_name_or_text.__name__, action_from_name
    )

    form = FormAction("my_form", endpoint_config)
    events = await form._ask_for_slot(
        Domain.empty(),
        default_nlg,
        CollectingOutputChannel(),
        "some slot",
        DialogueStateTracker.from_events("dasd", []),
    )

    assert not events
    action_from_name.assert_not_called()


@pytest.mark.parametrize(
    "ignored_intents, slot_not_intent",
    [
        # for entity_type -> from_entity
        (
            # `ignored_intents` as a string and slot's not_intent as an empty list.
            "greet",
            [],
        ),
        (
            # `ignored_intents` as an empty list and slot's not_intent has a value.
            [],
            ["greet"],
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value different than the ones in `ignored_intents`.
            ["chitchat", "greet"],
            ["inform"],
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value that is included also in `ignored_intents`.
            ["chitchat", "greet"],
            ["chitchat"],
        ),
    ],
)
async def test_ignored_intents_with_slot_type_from_entity(
    ignored_intents: Union[Text, List[Text]], slot_not_intent: Union[Text, List[Text]]
):
    form_name = "some_form"
    entity_name = "some_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "slots": {
                entity_name: {
                    "type": "any",
                    "mappings": [
                        {
                            "type": "from_entity",
                            "entity": entity_name,
                            "not_intent": slot_not_intent,
                        }
                    ],
                }
            },
            "forms": {
                form_name: {
                    IGNORED_INTENTS: ignored_intents,
                    REQUIRED_SLOTS_KEY: [entity_name],
                }
            },
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "hello",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(slot_events, domain)

    assert slot_events == []

    with pytest.raises(ActionExecutionRejection):
        await form.validate(
            tracker,
            domain,
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
        )


@pytest.mark.parametrize(
    "ignored_intents, slot_not_intent",
    [
        # same examples for entity_type -> from_text
        (
            # `ignored_intents` as a string and slot's not_intent as an empty list.
            "greet",
            [],
        ),
        (
            # `ignored_intents` as an empty list and slot's not_intent has a value.
            [],
            ["greet"],
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value different than the ones in `ignored_intents`.
            ["chitchat", "greet"],
            ["inform"],
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value that is included also in `ignored_intents`.
            ["chitchat", "greet"],
            ["chitchat"],
        ),
    ],
)
async def test_ignored_intents_with_slot_type_from_text(
    ignored_intents: Union[Text, List[Text]], slot_not_intent: Union[Text, List[Text]]
):
    form_name = "some_form"
    entity_name = "some_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "slots": {
                entity_name: {
                    "type": "any",
                    "mappings": [
                        {
                            "type": "from_text",
                            "intent": "some_intent",
                            "not_intent": slot_not_intent,
                        }
                    ],
                }
            },
            "forms": {
                form_name: {
                    IGNORED_INTENTS: ignored_intents,
                    REQUIRED_SLOTS_KEY: [entity_name],
                }
            },
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "hello",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(slot_events, domain)
    assert slot_events == []

    form_slot_events = await form.validate(
        tracker,
        domain,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
    )
    assert form_slot_events == []


@pytest.mark.parametrize(
    "ignored_intents, slot_not_intent, entity_type",
    [
        # same examples for entity_type -> from_intent
        (
            # `ignored_intents` as a string and slot's not_intent as an empty list.
            "greet",
            [],
            "from_intent",
        ),
        (
            # `ignored_intents` as an empty list and slot's not_intent has a value.
            [],
            ["greet"],
            "from_intent",
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value different than the ones in `ignored_intents`.
            ["chitchat", "greet"],
            ["inform"],
            "from_intent",
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value that is included also in `ignored_intents`.
            ["chitchat", "greet"],
            ["chitchat"],
            "from_intent",
        ),
        # same examples for entity_type -> from_trigger_intent
        (
            # `ignored_intents` as a string and slot's not_intent as an empty list.
            "greet",
            [],
            "from_trigger_intent",
        ),
        (
            # `ignored_intents` as an empty list and slot's not_intent has a value.
            [],
            ["greet"],
            "from_trigger_intent",
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value different than the ones in `ignored_intents`.
            ["chitchat", "greet"],
            ["inform"],
            "from_trigger_intent",
        ),
        (
            # `ignored_intents` as a list of 2 values and slot's not_intent has one
            # value that is included also in `ignored_intents`.
            ["chitchat", "greet"],
            ["chitchat"],
            "from_trigger_intent",
        ),
    ],
)
async def test_ignored_intents_with_other_type_of_slots(
    ignored_intents: Union[Text, List[Text]],
    slot_not_intent: Union[Text, List[Text]],
    entity_type: Text,
):
    form_name = "some_form"
    entity_name = "some_slot"
    form = FormAction(form_name, None)

    domain = Domain.from_dict(
        {
            "slots": {
                entity_name: {
                    "type": "any",
                    "mappings": [
                        {
                            "type": entity_type,
                            "value": "affirm",
                            "intent": "true",
                            "not_intent": slot_not_intent,
                        }
                    ],
                }
            },
            "forms": {
                form_name: {
                    IGNORED_INTENTS: ignored_intents,
                    REQUIRED_SLOTS_KEY: [entity_name],
                }
            },
        }
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            SlotSet(REQUESTED_SLOT, "some_slot"),
            UserUttered(
                "hello",
                intent={"name": "greet", "confidence": 1.0},
                entities=[{"entity": entity_name, "value": "some_value"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(slot_events, domain)
    assert slot_events == []

    form_slot_events = await form.validate(
        tracker,
        domain,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
    )
    assert form_slot_events == []


async def test_extract_other_slots_with_matched_mapping_conditions():
    form_name = "some_form"
    form = FormAction(form_name, None)

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intent:
            - greet
            - inform
            entities:
            - email
            - name
            slots:
              name:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: name
                  conditions:
                  - active_loop: some_form
                    requested_slot: email
              email:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: email
            forms:
             some_form:
               required_slots:
                 - email
                 - name
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "email"),
            UserUttered(
                "My name is Emily.",
                intent={"name": "inform", "confidence": 1.0},
                entities=[{"entity": "name", "value": "Emily"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert slot_events == [SlotSet("name", "Emily")]
    tracker.update_with_events(slot_events, domain)

    form_slot_events = await form.validate(
        tracker,
        domain,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
    )
    assert form_slot_events == []


async def test_extract_other_slots_raises_no_matched_conditions():
    form_name = "some_form"
    form = FormAction(form_name, None)

    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intent:
            - greet
            - inform
            entities:
            - email
            - name
            slots:
              name:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: name
                  conditions:
                  - active_loop: some_form
                    requested_slot: name
              email:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: email
            forms:
             some_form:
               required_slots:
                 - email
                 - name
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "email"),
            UserUttered(
                "My name is Emily.",
                intent={"name": "inform", "confidence": 1.0},
                entities=[{"entity": "name", "value": "Emily"}],
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )

    action_extract_slots = ActionExtractSlots(action_endpoint=None)

    slot_events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    tracker.update_with_events(slot_events, domain)

    assert slot_events == []

    with pytest.raises(ActionExecutionRejection):
        await form.validate(
            tracker,
            domain,
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
        )


async def test_action_extract_slots_custom_mapping_with_condition():
    domain_yaml = textwrap.dedent(
        f"""
        version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

        slots:
          custom_slot:
            type: text
            influence_conversation: false
            mappings:
            - type: custom
              conditions:
              - active_loop: some_form

        forms:
          some_form:
            required_slots:
            - custom_slot

        actions:
        - validate_some_form
        """
    )
    domain = Domain.from_yaml(domain_yaml)
    events = [ActiveLoop("some_form"), UserUttered("Hi")]
    tracker = DialogueStateTracker.from_events(
        sender_id="test_id", evts=events, slots=domain.slots
    )

    with aioresponses() as mocked:
        mocked.post(
            ACTION_SERVER_URL,
            payload={
                "events": [{"event": "slot", "name": "custom_slot", "value": "test"}]
            },
        )

        action_server = EndpointConfig(ACTION_SERVER_URL)
        action_extract_slots = ActionExtractSlots(action_server)
        events = await action_extract_slots.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert events == []

        form = FormAction("some_form", action_server)
        form_events = await form.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert form_events == [
            SlotSet("custom_slot", "test"),
            SlotSet(REQUESTED_SLOT, None),
            ActiveLoop(None),
        ]


async def test_form_slots_empty_with_restart():
    domain = Domain.from_yaml(
        textwrap.dedent(
            f"""
            version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
            intent:
            - greet
            - inform
            entities:
            - name
            slots:
              name:
                type: text
                influence_conversation: false
                mappings:
                - type: from_entity
                  entity: name
                  conditions:
                  - active_loop: some_form
            forms:
             some_form:
               required_slots:
                 - name
            """
        )
    )

    tracker = DialogueStateTracker.from_events(
        "default",
        [
            UserUttered("hi", intent={"name": "greet", "confidence": 1.0}),
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "name"),
            UserUttered(
                "My name is Emily.",
                intent={"name": "inform", "confidence": 1.0},
                entities=[{"entity": "name", "value": "Emily"}],
            ),
            SlotSet("name", "emily"),
            Restarted(),
            ActiveLoop("some_form"),
            SlotSet(REQUESTED_SLOT, "name"),
            UserUttered("hi", intent={"name": "greet", "confidence": 1.0}),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    action = FormAction("some_form", None)

    # FormAction execution is rejected because a slot was requested but none
    # were extracted (events before restart are not considered).
    with pytest.raises(ActionExecutionRejection):
        await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )


async def test_extract_slots_with_mapping_conditions_during_form_activation():
    slot_name = "city"
    entity_value = "London"
    entity_name = "location"

    form_name = "test_form"

    domain = Domain.from_yaml(
        f"""
    entities:
    - {entity_name}
    slots:
      {slot_name}:
        type: text
        mappings:
        - type: from_entity
          entity: {entity_name}
          conditions:
          - active_loop: {form_name}
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
            - {slot_name}
    """
    )

    events = [
        ActionExecuted("action_listen"),
        UserUttered(
            "I live in London",
            entities=[{"entity": entity_name, "value": entity_value}],
        ),
    ]
    tracker = DialogueStateTracker.from_events(
        sender_id="test", evts=events, domain=domain, slots=domain.slots
    )
    assert tracker.active_loop_name is None

    action_extract_slots = ActionExtractSlots(None)
    events = await action_extract_slots.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )
    assert events == []

    expected_events = [
        ActiveLoop(form_name),
        SlotSet(slot_name, entity_value),
        SlotSet(REQUESTED_SLOT, None),
        ActiveLoop(None),
    ]

    form_action = FormAction(form_name, None)
    form_events = await form_action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        tracker,
        domain,
    )

    assert form_events == expected_events


async def test_form_validation_happens_once(caplog: LogCaptureFixture):
    """
    Tests if form validation happens once instead of twice.
    Solves the bug presented in https://rasahq.atlassian.net/browse/ENG-117
    """
    tracker = DialogueStateTracker.from_events(sender_id="bla", evts=[])
    form_name = "my_form"
    action_server = EndpointConfig(ACTION_SERVER_URL)
    action = FormAction(form_name, action_server)
    slot_name = "num_people"
    domain = textwrap.dedent(
        f"""
    slots:
      {slot_name}:
        type: float
        mappings:
        - type: from_entity
          entity: number
    forms:
      {form_name}:
        {REQUIRED_SLOTS_KEY}:
        - {slot_name}
    responses:
      utter_ask_num_people:
      - text: "How many people?"
    actions:
      - validate_{form_name}
    """
    )
    domain = Domain.from_yaml(domain)
    form_validation_events = [
        {
            "event": "slot",
            "timestamp": None,
            "name": "num_people",
            "value": 5,
        }
    ]
    with aioresponses() as mocked, caplog.at_level(logging.DEBUG):
        mocked.post(
            ACTION_SERVER_URL,
            payload={"events": form_validation_events},
        )
        _ = await action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert (
            sum(
                [
                    1
                    for message in caplog.messages
                    if f"Calling action endpoint to run action 'validate_{form_name}'."
                    in message
                ]
            )
            == 1
        )


async def test_form_validation_happens_at_form_activation(caplog: LogCaptureFixture):
    """Test if form validation happens at form activation.

    The particular case is when a non-required slot for the form is also filled
    at form activation.

    Fixes the bug in https://rasahq.atlassian.net/browse/ATO-1104.
    """
    form = "help_form"
    action_server = EndpointConfig(ACTION_SERVER_URL)
    form_action = FormAction(form, action_server)

    entity_a = "device"
    entity_b = "account_type"
    entity_b_value = "bronze"

    context_slot = "device_type"
    required_slot = "send_sms"
    global_slot = "membership"

    domain = textwrap.dedent(
        f"""
    intents:
    - start_form
    entities:
    - {entity_a}
    - {entity_b}
    slots:
      {context_slot}:
        type: categorical
        influence_conversation: false
        initial_value: "mobile"
        values:
        - mobile
        - desktop
        - other
        mappings:
        - type: from_entity
          entity: {entity_a}
      {required_slot}:
        type: float
        influence_conversation: false
        mappings:
        - type: custom
      {global_slot}:
        type: text
        influence_conversation: false
        mappings:
          - type: from_entity
            entity: {entity_b}
    forms:
      {form}:
        {REQUIRED_SLOTS_KEY}:
        - {required_slot}
    responses:
      utter_ask_{required_slot}:
      - text: "Would you like to receive an SMS?"
    actions:
      - validate_{form}
    """
    )
    domain = Domain.from_yaml(domain)

    tracker = DialogueStateTracker.from_events(
        sender_id="test",
        evts=[
            UserUttered(
                "Can you help me with my bronze account ",
                intent={"name": "start_form", "confidence": 1.0},
                entities=[{"entity": entity_b, "value": entity_b_value}],
            ),
        ],
    )

    # Mock the custom validation action to update form required_slots to empty list
    form_validation_events = [
        {
            "event": "slot",
            "timestamp": None,
            "name": "requested_slot",
            "value": None,
        }
    ]
    with aioresponses() as mocked, caplog.at_level(logging.DEBUG):
        mocked.post(
            ACTION_SERVER_URL,
            payload={"events": form_validation_events},
        )
        events = await form_action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert (
            sum(
                [
                    1
                    for message in caplog.messages
                    if f"Calling action endpoint to run action 'validate_{form}'."
                    in message
                ]
            )
            == 1
        )

        assert events == [
            ActiveLoop(form),
            SlotSet(REQUESTED_SLOT, None),
            SlotSet(global_slot, entity_b_value),
            ActiveLoop(None),
        ]


async def test_form_validation_happens_at_form_activation_with_empty_required_slots(
    caplog: LogCaptureFixture,
):
    """Test if required_slots get updated dynamically at form activation.

    The particular case is when a form has an empty required_slots list.
    """
    form = "booking_form"
    action_server = EndpointConfig(ACTION_SERVER_URL)
    form_action = FormAction(form, action_server)

    dynamic_slot = "customer_name"
    bot_utterance = "What is your name?"

    domain = textwrap.dedent(
        f"""
    intents:
    - start_form

    slots:
      {dynamic_slot}:
        type: text
        influence_conversation: false
        mappings:
        - type: custom

    forms:
      {form}:
        {REQUIRED_SLOTS_KEY}: []
    responses:
      utter_ask_{dynamic_slot}:
      - text: "{bot_utterance}"
    actions:
      - validate_{form}
    """
    )
    domain = Domain.from_yaml(domain)

    tracker = DialogueStateTracker.from_events(
        sender_id="test",
        evts=[
            UserUttered(
                "I'd like to make a booking. ",
                intent={"name": "start_form", "confidence": 1.0},
            ),
        ],
    )

    # Mock the custom validation action to update form required_slots
    form_validation_events = [
        {
            "event": "slot",
            "timestamp": None,
            "name": "requested_slot",
            "value": dynamic_slot,
        }
    ]
    with aioresponses() as mocked, caplog.at_level(logging.DEBUG):
        mocked.post(
            ACTION_SERVER_URL,
            payload={"events": form_validation_events},
        )
        events = await form_action.run(
            CollectingOutputChannel(),
            TemplatedNaturalLanguageGenerator(domain.responses),
            tracker,
            domain,
        )
        assert (
            sum(
                [
                    1
                    for message in caplog.messages
                    if f"Calling action endpoint to run action 'validate_{form}'."
                    in message
                ]
            )
            == 1
        )

        assert len(events) == 3

        assert events[0] == ActiveLoop(form)
        assert events[1] == SlotSet(REQUESTED_SLOT, dynamic_slot)
        assert isinstance(events[2], BotUttered) is True
        assert events[2].text == bot_utterance
