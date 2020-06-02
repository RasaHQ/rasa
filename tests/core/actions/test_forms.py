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
