import asyncio
import logging

import datetime
import pytest
import time
import uuid
import json
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses
from typing import Optional, Text
from unittest.mock import patch

from rasa.core import jobs
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME

from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from rasa.core.domain import SessionConfig
from rasa.core.events import (
    ActionExecuted,
    BotUttered,
    ReminderCancelled,
    ReminderScheduled,
    Restarted,
    UserUttered,
    SessionStarted,
    Event,
    SlotSet,
)
from rasa.core.interpreter import RasaNLUHttpInterpreter
from rasa.core.processor import MessageProcessor, DEFAULT_INTENTS
from rasa.core.slots import Slot
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import latest_request

logger = logging.getLogger(__name__)


async def test_message_processor(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    await default_processor.handle_message(
        UserMessage('/greet{"name":"Core"}', default_channel)
    )
    assert {
        "recipient_id": "default",
        "text": "hey there Core!",
    } == default_channel.latest_output()


async def test_message_id_logging(default_processor: MessageProcessor):
    from rasa.core.trackers import DialogueStateTracker

    message = UserMessage("If Meg was an egg would she still have a leg?")
    tracker = DialogueStateTracker("1", [])
    await default_processor._handle_message_with_tracker(message, tracker)
    logged_event = tracker.events[-1]

    assert logged_event.message_id == message.message_id
    assert logged_event.message_id is not None


async def test_parsing(default_processor: MessageProcessor):
    message = UserMessage('/greet{"name": "boy"}')
    parsed = await default_processor._parse_message(message)
    assert parsed["intent"]["name"] == "greet"
    assert parsed["entities"][0]["entity"] == "name"


async def test_log_unseen_feature(default_processor: MessageProcessor):
    message = UserMessage('/dislike{"test_entity": "RASA"}')
    parsed = await default_processor._parse_message(message)
    with pytest.warns(UserWarning) as record:
        default_processor._log_unseen_features(parsed)
    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "Interpreter parsed an intent 'dislike' that is not defined in the domain."
    )
    assert (
        record[1].message.args[0]
        == "Interpreter parsed an entity 'test_entity' that is not defined in the domain."
    )


@pytest.mark.parametrize("default_intent", DEFAULT_INTENTS)
async def test_default_intent_recognized(
    default_processor: MessageProcessor, default_intent: Text
):
    message = UserMessage(default_intent)
    parsed = await default_processor._parse_message(message)
    with pytest.warns(None) as record:
        default_processor._log_unseen_features(parsed)
    assert len(record) == 0


async def test_http_parsing():
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")
    with aioresponses() as mocked:
        mocked.post("https://interpreter.com/model/parse", repeat=True, status=200)

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        try:
            await MessageProcessor(inter, None, None, None, None)._parse_message(
                message
            )
        except KeyError:
            pass  # logger looks for intent and entities, so we except

        r = latest_request(mocked, "POST", "https://interpreter.com/model/parse")

        assert r


async def mocked_parse(self, text, message_id=None, tracker=None):
    """Mock parsing a text message and augment it with the slot
    value from the tracker's state."""

    return {
        "intent": {"name": "", "confidence": 0.0},
        "entities": [],
        "text": text,
        "requested_language": tracker.get_slot("requested_language"),
    }


async def test_parsing_with_tracker():
    tracker = DialogueStateTracker.from_dict("1", [], [Slot("requested_language")])

    # we'll expect this value 'en' to be part of the result from the interpreter
    tracker._set_slot("requested_language", "en")

    endpoint = EndpointConfig("https://interpreter.com")
    with aioresponses() as mocked:
        mocked.post("https://interpreter.com/parse", repeat=True, status=200)

        # mock the parse function with the one defined for this test
        with patch.object(RasaNLUHttpInterpreter, "parse", mocked_parse):
            interpreter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
            agent = Agent(None, None, interpreter)
            result = await agent.parse_message_using_nlu_interpreter("lunch?", tracker)

            assert result["requested_language"] == "en"


async def test_reminder_scheduled(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled("utter_greet", datetime.datetime.now())
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(UserUttered("test"))
    tracker.update(ActionExecuted("action_reminder_reminder"))
    tracker.update(reminder)

    default_processor.tracker_store.save(tracker)

    await default_processor.handle_reminder(
        reminder, sender_id, default_channel, default_processor.nlg
    )

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)

    assert t.events[-4] == UserUttered(None)
    assert t.events[-3] == ActionExecuted("utter_greet")
    assert t.events[-2] == BotUttered(
        "hey there None!",
        {
            "elements": None,
            "buttons": None,
            "quick_replies": None,
            "attachment": None,
            "image": None,
            "custom": None,
        },
    )
    assert t.events[-1] == ActionExecuted("action_listen")


async def test_reminder_aborted(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=True
    )
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(reminder)
    tracker.update(UserUttered("test"))  # cancels the reminder

    default_processor.tracker_store.save(tracker)
    await default_processor.handle_reminder(
        reminder, sender_id, default_channel, default_processor.nlg
    )

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 3  # nothing should have been executed


async def test_reminder_cancelled(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    trackers = []
    for sender_id in sender_ids:
        tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_reminder_reminder"))
        tracker.update(
            ReminderScheduled(
                "utter_greet", datetime.datetime.now(), kill_on_user_message=True
            )
        )
        trackers.append(tracker)

    # cancel reminder for the first user
    trackers[0].update(ReminderCancelled("utter_greet"))

    for tracker in trackers:
        default_processor.tracker_store.save(tracker)
        await default_processor._schedule_reminders(
            tracker.events, tracker, default_channel, default_processor.nlg
        )
    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == 2

    for tracker in trackers:
        await default_processor._cancel_reminders(tracker.events, tracker)
    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == 1

    # execute the jobs
    await asyncio.sleep(5)

    tracker_0 = default_processor.tracker_store.retrieve(sender_ids[0])
    # there should be no utter_greet action
    assert ActionExecuted("utter_greet") not in tracker_0.events

    tracker_1 = default_processor.tracker_store.retrieve(sender_ids[1])
    # there should be utter_greet action
    assert ActionExecuted("utter_greet") in tracker_1.events


async def test_reminder_restart(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=False
    )
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(reminder)
    tracker.update(Restarted())  # cancels the reminder
    tracker.update(UserUttered("test"))

    default_processor.tracker_store.save(tracker)
    await default_processor.handle_reminder(
        reminder, sender_id, default_channel, default_processor.nlg
    )

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 4  # nothing should have been executed


@pytest.mark.parametrize(
    "event_to_apply,session_expiration_time_in_minutes,has_expired",
    [
        # last user event is way in the past
        (UserUttered(timestamp=1), 60, True),
        # user event are very recent
        (UserUttered("hello", timestamp=time.time()), 60, False,),
        # there is user event
        (ActionExecuted(ACTION_LISTEN_NAME, timestamp=time.time()), 60, False),
        # Old event, but sessions are disabled
        (UserUttered("hello", timestamp=1), 0, False),
        # there is no event
        (None, 1, False),
    ],
)
async def test_has_session_expired(
    event_to_apply: Optional[Event],
    session_expiration_time_in_minutes: float,
    has_expired: bool,
    default_processor: MessageProcessor,
):
    sender_id = uuid.uuid4().hex

    default_processor.domain.session_config = SessionConfig(
        session_expiration_time_in_minutes, True
    )
    # create new tracker without events
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)
    tracker.events.clear()

    # apply desired event
    if event_to_apply:
        tracker.update(event_to_apply)

    # noinspection PyProtectedMember
    assert default_processor._has_session_expired(tracker) == has_expired


# noinspection PyProtectedMember
async def test_update_tracker_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    # patch `_has_session_expired()` so the `_update_tracker_session()` call actually
    # does something
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    await default_processor._update_tracker_session(tracker, default_channel)

    # the save is not called in _update_tracker_session()
    default_processor._save_tracker(tracker)

    # inspect tracker and make sure all events are present
    tracker = default_processor.tracker_store.retrieve(sender_id)

    assert list(tracker.events) == [
        ActionExecuted(ACTION_LISTEN_NAME),
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]


# noinspection PyProtectedMember
async def test_update_tracker_session_with_slots(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    # apply a user uttered and five slots
    user_event = UserUttered("some utterance")
    tracker.update(user_event)

    slot_set_events = [SlotSet(f"slot key {i}", f"test value {i}") for i in range(5)]

    for event in slot_set_events:
        tracker.update(event)

    # patch `_has_session_expired()` so the `_update_tracker_session()` call actually
    # does something
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    await default_processor._update_tracker_session(tracker, default_channel)

    # the save is not called in _update_tracker_session()
    default_processor._save_tracker(tracker)

    # inspect tracker and make sure all events are present
    tracker = default_processor.tracker_store.retrieve(sender_id)
    events = list(tracker.events)

    # the first three events should be up to the user utterance
    assert events[:2] == [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_event,
    ]

    # next come the five slots
    assert events[2:7] == slot_set_events

    # the next two events are the session start sequence
    assert events[7:9] == [ActionExecuted(ACTION_SESSION_START_NAME), SessionStarted()]

    # the five slots should be reapplied
    assert events[9:14] == slot_set_events

    # finally an action listen, this should also be the last event
    assert events[14] == events[-1] == ActionExecuted(ACTION_LISTEN_NAME)


# noinspection PyProtectedMember
async def test_get_tracker_with_session_start(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor,
):
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.get_tracker_with_session_start(
        sender_id, default_channel
    )

    # ensure session start sequence is present
    assert list(tracker.events) == [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]


async def test_handle_message_with_session_start(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex

    entity = "name"
    slot_1 = {entity: "Core"}
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_1)}", default_channel, sender_id)
    )

    assert {
        "recipient_id": sender_id,
        "text": "hey there Core!",
    } == default_channel.latest_output()

    # patch processor so a session start is triggered
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    slot_2 = {entity: "post-session start hello"}
    # handle a new message
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_2)}", default_channel, sender_id)
    )

    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    # make sure the sequence of events is as expected
    assert list(tracker.events) == [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            f"/greet{json.dumps(slot_1)}",
            {"name": "greet", "confidence": 1.0},
            [{"entity": entity, "start": 6, "end": 22, "value": "Core"}],
        ),
        SlotSet(entity, slot_1[entity]),
        ActionExecuted("utter_greet"),
        BotUttered("hey there Core!"),
        ActionExecuted(ACTION_LISTEN_NAME),
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        # the initial SlotSet is reapplied after the SessionStarted sequence
        SlotSet(entity, slot_1[entity]),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            f"/greet{json.dumps(slot_2)}",
            {"name": "greet", "confidence": 1.0},
            [
                {
                    "entity": entity,
                    "start": 6,
                    "end": 42,
                    "value": "post-session start hello",
                }
            ],
        ),
        SlotSet(entity, slot_2[entity]),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]


# noinspection PyProtectedMember
@pytest.mark.parametrize(
    "action_name, should_predict_another_action",
    [
        (ACTION_LISTEN_NAME, False),
        (ACTION_SESSION_START_NAME, False),
        ("utter_greet", True),
    ],
)
async def test_should_predict_another_action(
    default_processor: MessageProcessor,
    action_name: Text,
    should_predict_another_action: bool,
):
    assert (
        default_processor.should_predict_another_action(action_name)
        == should_predict_another_action
    )
