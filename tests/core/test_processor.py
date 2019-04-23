import datetime
import pytest
import uuid
from aioresponses import aioresponses

import asyncio
import rasa.utils.io
from rasa.core import jobs
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.dispatcher import Button, Dispatcher
from rasa.core.events import (
    ReminderScheduled,
    ReminderCancelled,
    UserUttered,
    ActionExecuted,
    BotUttered,
    Restarted,
)
from rasa.core.processor import MessageProcessor
from rasa.core.interpreter import RasaNLUHttpInterpreter
from rasa.utils.endpoints import EndpointConfig

from tests.utilities import json_of_latest_request, latest_request


@pytest.fixture(scope="module")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop

    return rasa.utils.io.enable_async_loop_debugging(next(sanic_loop()))


async def test_message_processor(default_processor):
    out = CollectingOutputChannel()
    await default_processor.handle_message(UserMessage('/greet{"name":"Core"}', out))
    assert {"recipient_id": "default", "text": "hey there Core!"} == out.latest_output()


async def test_message_id_logging(default_processor):
    from rasa.core.trackers import DialogueStateTracker

    message = UserMessage("If Meg was an egg would she still have a leg?")
    tracker = DialogueStateTracker("1", [])
    await default_processor._handle_message_with_tracker(message, tracker)
    logged_event = tracker.events[-1]

    assert logged_event.message_id == message.message_id
    assert logged_event.message_id is not None


async def test_parsing(default_processor):
    message = UserMessage('/greet{"name": "boy"}')
    parsed = await default_processor._parse_message(message)
    assert parsed["intent"]["name"] == "greet"
    assert parsed["entities"][0]["entity"] == "name"


async def test_http_parsing():
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")
    with aioresponses() as mocked:
        mocked.post("https://interpreter.com/parse", repeat=True, status=200)

        inter = RasaNLUHttpInterpreter(endpoint=endpoint)
        try:
            await MessageProcessor(inter, None, None, None, None)._parse_message(
                message
            )
        except KeyError:
            pass  # logger looks for intent and entities, so we except

        r = latest_request(mocked, "POST", "https://interpreter.com/parse")

        assert r
        assert json_of_latest_request(r)["message_id"] == message.message_id


async def test_reminder_scheduled(default_processor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled("utter_greet", datetime.datetime.now())
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(UserUttered("test"))
    t.update(ActionExecuted("action_reminder_reminder"))
    t.update(r)

    default_processor.tracker_store.save(t)
    await default_processor.handle_reminder(r, d)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert t.events[-4] == UserUttered(None)
    assert t.events[-3] == ActionExecuted("utter_greet")
    assert t.events[-2] == BotUttered(
        "hey there None!", {"elements": None, "buttons": None, "attachment": None}
    )
    assert t.events[-1] == ActionExecuted("action_listen")


async def test_reminder_aborted(default_processor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=True
    )
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(r)
    t.update(UserUttered("test"))  # cancels the reminder

    default_processor.tracker_store.save(t)
    await default_processor.handle_reminder(r, d)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 3  # nothing should have been executed


async def test_reminder_cancelled(default_processor):
    out = CollectingOutputChannel()
    sender_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    trackers = []
    for sender_id in sender_ids:
        t = default_processor.tracker_store.get_or_create_tracker(sender_id)

        t.update(UserUttered("test"))
        t.update(ActionExecuted("action_reminder_reminder"))
        t.update(
            ReminderScheduled(
                "utter_greet", datetime.datetime.now(), kill_on_user_message=True
            )
        )
        trackers.append(t)

    # cancel reminder for the first user
    trackers[0].update(ReminderCancelled("utter_greet"))

    for i, t in enumerate(trackers):
        default_processor.tracker_store.save(t)
        d = Dispatcher(sender_id, out, default_processor.nlg)
        await default_processor._schedule_reminders(t.events, t, d)
    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == 2

    for t in trackers:
        await default_processor._cancel_reminders(t.events, t)
    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == 1

    # execute the jobs
    await asyncio.sleep(3)

    t = default_processor.tracker_store.retrieve(sender_ids[0])
    # there should be no utter_greet action
    assert ActionExecuted("utter_greet") not in t.events

    t = default_processor.tracker_store.retrieve(sender_ids[1])
    # there should be utter_greet action
    assert ActionExecuted("utter_greet") in t.events


async def test_reminder_restart(default_processor: MessageProcessor):
    out = CollectingOutputChannel()
    sender_id = uuid.uuid4().hex

    d = Dispatcher(sender_id, out, default_processor.nlg)
    r = ReminderScheduled(
        "utter_greet", datetime.datetime.now(), kill_on_user_message=False
    )
    t = default_processor.tracker_store.get_or_create_tracker(sender_id)

    t.update(r)
    t.update(Restarted())  # cancels the reminder
    t.update(UserUttered("test"))

    default_processor.tracker_store.save(t)
    await default_processor.handle_reminder(r, d)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 4  # nothing should have been executed


async def test_logging_of_bot_utterances_on_tracker(
    default_processor, default_dispatcher_collecting, default_agent
):
    sender_id = "test_logging_of_bot_utterances_on_tracker"
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)
    buttons = [
        Button(title="Btn1", payload="_btn1"),
        Button(title="Btn2", payload="_btn2"),
    ]

    await default_dispatcher_collecting.utter_template("utter_goodbye", tracker)
    await default_dispatcher_collecting.utter_attachment("http://my-attachment")
    await default_dispatcher_collecting.utter_message("my test message")
    await default_dispatcher_collecting.utter_button_message("my message", buttons)

    assert len(default_dispatcher_collecting.latest_bot_messages) == 4

    default_processor.log_bot_utterances_on_tracker(
        tracker, default_dispatcher_collecting
    )
    assert not default_dispatcher_collecting.latest_bot_messages
