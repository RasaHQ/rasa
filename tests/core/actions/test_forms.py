from aioresponses import aioresponses

from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.actions.forms import FormAction, REQUESTED_SLOT
from rasa.core.domain import Domain
from rasa.core.events import Form, SlotSet, UserUttered, ActionExecuted
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
"""
    domain = Domain.from_yaml(domain)
    events = await action.run(None, None, tracker, domain)
    assert events == [Form(form_name), SlotSet(REQUESTED_SLOT, slot_name)]


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
    """
    domain = Domain.from_yaml(domain)

    action = FormAction(form_name, None)
    events = await action.run(None, None, tracker, domain)
    assert events == [
        SlotSet(slot_name, slot_value),
        Form(None),
        SlotSet(REQUESTED_SLOT, None),
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
    forms:
    - {form_name}:
        {slot_name}:
        - type: from_text
    actions:
    - action_validate_{form_name}
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

        events = await action.run(None, None, tracker, domain)
        assert events == [
            SlotSet(slot_name, validated_slot_value),
            Form(None),
            SlotSet(REQUESTED_SLOT, None),
        ]
