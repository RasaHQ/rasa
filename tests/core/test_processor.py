import asyncio
import datetime
import uuid

import pytest
from aioresponses import aioresponses

import rasa.utils.io
from rasa.core import jobs
from rasa.core.channels import CollectingOutputChannel, UserMessage
from rasa.core.events import (
    ActionExecuted,
    BotUttered,
    ReminderCancelled,
    ReminderScheduled,
    Restarted,
    UserUttered,
)
from rasa.core.interpreter import RasaNLUHttpInterpreter
from rasa.core.processor import MessageProcessor
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import json_of_latest_request, latest_request


@pytest.fixture(scope="session")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


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
    await asyncio.sleep(3)

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
