import asyncio
import datetime
from http import HTTPStatus
import textwrap
from pathlib import Path

import freezegun
import pytest
import time
import uuid
import json
from _pytest.monkeypatch import MonkeyPatch
from _pytest.logging import LogCaptureFixture
from aioresponses import aioresponses
from typing import Optional, Text, List, Callable, Type, Any

from rasa.core.lock_store import InMemoryLockStore
from rasa.core.policies.ensemble import DefaultPolicyPredictionEnsemble
from rasa.core.tracker_store import InMemoryTrackerStore
import rasa.shared.utils.io
from rasa.core.actions.action import (
    ActionBotResponse,
    ActionListen,
    ActionExecutionRejection,
    ActionUnlikelyIntent,
)
from rasa.core.nlg import NaturalLanguageGenerator, TemplatedNaturalLanguageGenerator
from rasa.core.policies.policy import PolicyPrediction
from tests.conftest import with_model_id, with_model_ids
import tests.utilities

from rasa.core import jobs
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import (
    CollectingOutputChannel,
    UserMessage,
    OutputChannel,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.exceptions import ActionLimitReached
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import SessionConfig, Domain, KEY_ACTIONS
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    ReminderCancelled,
    ReminderScheduled,
    Restarted,
    UserUttered,
    SessionStarted,
    Event,
    SlotSet,
    DefinePrevUserUtteredFeaturization,
    ActionExecutionRejected,
    LoopInterrupted,
)
from rasa.core.http_interpreter import RasaNLUHttpInterpreter
from rasa.core.processor import MessageProcessor
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import INTENT_NAME_KEY, METADATA_MODEL_ID
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.endpoints import EndpointConfig
from rasa.shared.core.constants import (
    ACTION_RESTART_NAME,
    ACTION_UNLIKELY_INTENT_NAME,
    DEFAULT_INTENTS,
    ACTION_LISTEN_NAME,
    ACTION_SESSION_START_NAME,
    EXTERNAL_MESSAGE_PREFIX,
    IS_EXTERNAL,
    SESSION_START_METADATA_SLOT,
)

import logging

logger = logging.getLogger(__name__)


async def test_message_processor(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    await default_processor.handle_message(
        UserMessage('/greet{"name":"Core"}', default_channel)
    )
    assert default_channel.latest_output() == {
        "recipient_id": "default",
        "text": "hey there Core!",
    }


async def test_message_id_logging(default_processor: MessageProcessor):
    message = UserMessage("If Meg was an egg would she still have a leg?")
    tracker = DialogueStateTracker("1", [])
    await default_processor._handle_message_with_tracker(message, tracker)
    logged_event = tracker.events[-1]

    assert logged_event.message_id == message.message_id
    assert logged_event.message_id is not None


async def test_parsing(default_processor: MessageProcessor):
    message = UserMessage('/greet{"name": "boy"}')
    parsed = await default_processor.parse_message(message)
    assert parsed["intent"][INTENT_NAME_KEY] == "greet"
    assert parsed["entities"][0]["entity"] == "name"


async def test_check_for_unseen_feature(default_processor: MessageProcessor):
    message = UserMessage('/greet{"name": "Joe"}')
    old_domain = default_processor.domain
    new_domain = Domain.from_dict(old_domain.as_dict())
    new_domain.intent_properties = {
        name: intent
        for name, intent in new_domain.intent_properties.items()
        if name != "greet"
    }
    new_domain.entities = [e for e in new_domain.entities if e != "name"]
    default_processor.domain = new_domain

    parsed = await default_processor.parse_message(message)
    with pytest.warns(UserWarning) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 2

    assert record[0].message.args[0].startswith("Parsed an intent 'greet'")
    assert record[1].message.args[0].startswith("Parsed an entity 'name'")

    default_processor.domain = old_domain


@pytest.mark.parametrize("default_intent", DEFAULT_INTENTS)
async def test_default_intent_recognized(
    default_processor: MessageProcessor, default_intent: Text
):
    message = UserMessage(f"/{default_intent}")
    parsed = await default_processor.parse_message(message)
    with pytest.warns(None) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 0


async def test_http_parsing(trained_default_agent_model: Text, domain: Domain):
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")

    response_body = {
        "intent": {INTENT_NAME_KEY: "some_intent", "confidence": 1.0},
        "entities": [],
        "text": "lunch?",
    }

    with aioresponses() as mocked:
        mocked.post(
            "https://interpreter.com/model/parse",
            repeat=True,
            status=HTTPStatus.OK,
            body=json.dumps(response_body),
        )

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        processor = MessageProcessor(
            trained_default_agent_model,
            InMemoryTrackerStore(domain),
            InMemoryLockStore(),
            NaturalLanguageGenerator(),
            http_interpreter=inter,
        )
        data = await processor.parse_message(message)

        r = tests.utilities.latest_request(
            mocked, "POST", "https://interpreter.com/model/parse"
        )

        assert r
        assert data == response_body


async def test_http_parsing_default_response(
    trained_default_agent_model: Text, domain: Domain
):
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")

    with aioresponses() as mocked:
        mocked.post(
            "https://interpreter.com/model/parse",
            repeat=True,
            status=HTTPStatus.OK,
            body=None,
        )

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        processor = MessageProcessor(
            trained_default_agent_model,
            InMemoryTrackerStore(domain),
            InMemoryLockStore(),
            NaturalLanguageGenerator(),
            http_interpreter=inter,
        )
        data = await processor.parse_message(message)

        r = tests.utilities.latest_request(
            mocked, "POST", "https://interpreter.com/model/parse"
        )

        assert r
        assert data == {
            "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
            "entities": [],
            "text": "",
        }


async def test_reminder_scheduled(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex

    reminder = ReminderScheduled("remind", datetime.datetime.now())
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(UserUttered("test"))
    tracker.update(ActionExecuted("action_schedule_reminder"))
    tracker.update(reminder)

    default_processor.tracker_store.save(tracker)

    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)

    assert t.events[1] == UserUttered("test")
    assert t.events[2] == ActionExecuted("action_schedule_reminder")
    assert isinstance(t.events[3], ReminderScheduled)
    assert t.events[4] == UserUttered(
        f"{EXTERNAL_MESSAGE_PREFIX}remind",
        intent={INTENT_NAME_KEY: "remind", IS_EXTERNAL: True},
    )


async def test_reminder_lock(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    caplog: LogCaptureFixture,
):
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        sender_id = uuid.uuid4().hex

        reminder = ReminderScheduled("remind", datetime.datetime.now())
        tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

        tracker.update(UserUttered("test"))
        tracker.update(ActionExecuted("action_schedule_reminder"))
        tracker.update(reminder)

        default_processor.tracker_store.save(tracker)

        await default_processor.handle_reminder(reminder, sender_id, default_channel)

        assert f"Deleted lock for conversation '{sender_id}'." in caplog.text


async def test_trigger_external_latest_input_channel(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = uuid.uuid4().hex
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)
    input_channel = "test_input_channel_external"

    tracker.update(UserUttered("test1"))
    tracker.update(UserUttered("test2", input_channel=input_channel))

    await default_processor.trigger_external_user_uttered(
        "test3", None, tracker, default_channel
    )

    tracker = default_processor.tracker_store.retrieve(sender_id)

    assert tracker.get_latest_input_channel() == input_channel


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
    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 3  # nothing should have been executed


async def wait_until_all_jobs_were_executed(
    timeout_after_seconds: Optional[float] = None,
) -> None:
    total_seconds = 0.0
    while len((await jobs.scheduler()).get_jobs()) > 0 and (
        not timeout_after_seconds or total_seconds < timeout_after_seconds
    ):
        await asyncio.sleep(0.1)
        total_seconds += 0.1

    if total_seconds >= timeout_after_seconds:
        jobs.kill_scheduler()
        raise TimeoutError


async def test_reminder_cancelled_multi_user(
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
                "greet", datetime.datetime.now(), kill_on_user_message=True
            )
        )
        trackers.append(tracker)

    # cancel all reminders (one) for the first user
    trackers[0].update(ReminderCancelled())

    for tracker in trackers:
        default_processor.tracker_store.save(tracker)
        await default_processor._schedule_reminders(
            tracker.events, tracker, default_channel
        )
    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == 2

    for tracker in trackers:
        await default_processor._cancel_reminders(tracker.events, tracker)
    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == 1

    # execute the jobs
    await wait_until_all_jobs_were_executed(timeout_after_seconds=5.0)

    tracker_0 = default_processor.tracker_store.retrieve(sender_ids[0])
    # there should be no utter_greet action
    assert (
        UserUttered(
            f"{EXTERNAL_MESSAGE_PREFIX}greet",
            intent={INTENT_NAME_KEY: "greet", IS_EXTERNAL: True},
        )
        not in tracker_0.events
    )

    tracker_1 = default_processor.tracker_store.retrieve(sender_ids[1])
    # there should be utter_greet action
    assert (
        UserUttered(
            f"{EXTERNAL_MESSAGE_PREFIX}greet",
            intent={INTENT_NAME_KEY: "greet", IS_EXTERNAL: True},
        )
        in tracker_1.events
    )


async def test_reminder_cancelled_cancels_job_with_name(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = "][]][xy,,=+2f'[:/;>]  <0d]A[e_,02"

    reminder = ReminderScheduled(
        intent="greet", trigger_date_time=datetime.datetime.now()
    )
    job_name = reminder.scheduled_job_name(sender_id)
    reminder_cancelled = ReminderCancelled()

    assert reminder_cancelled.cancels_job_with_name(job_name, sender_id)
    assert not reminder_cancelled.cancels_job_with_name(job_name.upper(), sender_id)


async def test_reminder_cancelled_cancels_job_with_name_special_name(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    sender_id = "][]][xy,,=+2f'[:/;  >]<0d]A[e_,02"
    name = "wkjbgr,34(,*&%^^&*(OP#LKMN V#NF# # #R"

    reminder = ReminderScheduled(
        intent="greet", trigger_date_time=datetime.datetime.now(), name=name
    )
    job_name = reminder.scheduled_job_name(sender_id)
    reminder_cancelled = ReminderCancelled(name)

    assert reminder_cancelled.cancels_job_with_name(job_name, sender_id)
    assert not reminder_cancelled.cancels_job_with_name(job_name.upper(), sender_id)


async def cancel_reminder_and_check(
    tracker: DialogueStateTracker,
    default_processor: MessageProcessor,
    reminder_canceled_event: ReminderCancelled,
    num_jobs_before: int,
    num_jobs_after: int,
) -> None:
    # cancel the sixth reminder
    tracker.update(reminder_canceled_event)

    # check that the jobs were added
    assert len((await jobs.scheduler()).get_jobs()) == num_jobs_before

    await default_processor._cancel_reminders(tracker.events, tracker)

    # check that only one job was removed
    assert len((await jobs.scheduler()).get_jobs()) == num_jobs_after


async def test_reminder_cancelled_by_name(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the sixth reminder
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled("special"), 6, 5
    )


async def test_reminder_cancelled_by_entities(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the fourth reminder
    await cancel_reminder_and_check(
        tracker,
        default_processor,
        ReminderCancelled(entities=[{"entity": "name", "value": "Bruce Wayne"}]),
        6,
        5,
    )


async def test_reminder_cancelled_by_intent(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel the third, fifth, and sixth reminder
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled(intent="default"), 6, 3
    )


async def test_reminder_cancelled_all(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    tracker_with_six_scheduled_reminders: DialogueStateTracker,
):
    tracker = tracker_with_six_scheduled_reminders
    await default_processor._schedule_reminders(
        tracker.events, tracker, default_channel
    )

    # cancel all reminders
    await cancel_reminder_and_check(
        tracker, default_processor, ReminderCancelled(), 6, 0
    )


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
    await default_processor.handle_reminder(reminder, sender_id, default_channel)

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)
    assert len(t.events) == 4  # nothing should have been executed


@pytest.mark.parametrize(
    "event_to_apply,session_expiration_time_in_minutes,has_expired",
    [
        # last user event is way in the past
        (UserUttered(timestamp=1), 60, True),
        # user event are very recent
        (UserUttered("hello", timestamp=time.time()), 120, False),
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


async def test_update_tracker_session_with_metadata(
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    model_id = default_processor.model_metadata.model_id
    sender_id = uuid.uuid4().hex
    message_metadata = {"metadataTestKey": "metadataTestValue"}
    message = UserMessage(
        text="hi",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        metadata=message_metadata,
    )
    await default_processor.handle_message(message)

    tracker = default_processor.tracker_store.retrieve(sender_id)
    events = list(tracker.events)

    assert events[0] == with_model_id(
        SlotSet(
            SESSION_START_METADATA_SLOT,
            message_metadata,
        ),
        model_id,
    )
    assert tracker.slots[SESSION_START_METADATA_SLOT].value == message_metadata

    assert events[1] == with_model_id(
        ActionExecuted(ACTION_SESSION_START_NAME), model_id=model_id
    )

    assert events[2] == with_model_id(SessionStarted(), model_id=model_id)
    assert events[2].metadata == {METADATA_MODEL_ID: model_id}
    assert events[3] == with_model_id(
        SlotSet(SESSION_START_METADATA_SLOT, message_metadata), model_id=model_id
    )
    assert events[4] == with_model_id(
        ActionExecuted(ACTION_LISTEN_NAME), model_id=model_id
    )

    assert isinstance(events[5], UserUttered)


@freezegun.freeze_time("2020-02-01")
async def test_custom_action_session_start_with_metadata(
    default_processor: MessageProcessor,
):
    domain = Domain.from_dict({KEY_ACTIONS: [ACTION_SESSION_START_NAME]})
    default_processor.domain = domain
    model_id = default_processor.model_metadata.model_id
    action_server_url = "http://some-url"
    default_processor.action_endpoint = EndpointConfig(action_server_url)

    sender_id = uuid.uuid4().hex
    metadata = {"metadataTestKey": "metadataTestValue"}
    message = UserMessage(
        text="hi",
        output_channel=CollectingOutputChannel(),
        sender_id=sender_id,
        metadata=metadata,
    )

    with aioresponses() as mocked:
        mocked.post(action_server_url, payload={"events": []})
        await default_processor.handle_message(message)

    last_request = tests.utilities.latest_request(mocked, "post", action_server_url)
    tracker_for_custom_action = tests.utilities.json_of_latest_request(last_request)[
        "tracker"
    ]

    assert tracker_for_custom_action["events"] == [
        {
            "event": "slot",
            "timestamp": 1580515200.0,
            "name": SESSION_START_METADATA_SLOT,
            "value": metadata,
            "metadata": {"model_id": model_id},
        }
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
    assert events[:2] == [ActionExecuted(ACTION_LISTEN_NAME), user_event]

    # next come the five slots
    assert events[2:7] == slot_set_events

    # the next two events are the session start sequence
    assert events[7:9] == [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
    ]
    assert events[9:14] == slot_set_events

    # finally an action listen, this should also be the last event
    assert events[14] == events[-1] == ActionExecuted(ACTION_LISTEN_NAME)


async def test_fetch_tracker_and_update_session(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
):
    model_id = default_processor.model_metadata.model_id
    sender_id = uuid.uuid4().hex
    tracker = await default_processor.fetch_tracker_and_update_session(
        sender_id, default_channel
    )

    # ensure session start sequence is present
    assert list(tracker.events) == with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )


@pytest.mark.parametrize(
    "initial_events,expected_event_types",
    [
        # tracker is initially not empty - when it is fetched, it will just contain
        # these four events
        (
            [
                ActionExecuted(ACTION_SESSION_START_NAME),
                SessionStarted(),
                ActionExecuted(ACTION_LISTEN_NAME),
                UserUttered("/greet", {INTENT_NAME_KEY: "greet", "confidence": 1.0}),
            ],
            [ActionExecuted, SessionStarted, ActionExecuted, UserUttered],
        ),
        # tracker is initially empty, and contains the session start sequence when
        # fetched
        ([], [ActionExecuted, SessionStarted, ActionExecuted]),
    ],
)
async def test_fetch_tracker_with_initial_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    initial_events: List[Event],
    expected_event_types: List[Type[Event]],
):
    conversation_id = uuid.uuid4().hex

    tracker = DialogueStateTracker.from_events(conversation_id, initial_events)

    default_processor.tracker_store.save(tracker)

    tracker = await default_processor.fetch_tracker_with_initial_session(
        conversation_id, default_channel
    )

    # the events in the fetched tracker are as expected
    assert len(tracker.events) == len(expected_event_types)

    assert all(
        isinstance(tracker_event, expected_event_type)
        for tracker_event, expected_event_type in zip(
            tracker.events, expected_event_types
        )
    )


async def test_fetch_tracker_with_initial_session_does_not_update_session(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    conversation_id = uuid.uuid4().hex

    # the domain has a session expiration time of one second
    monkeypatch.setattr(
        default_processor.tracker_store.domain,
        "session_config",
        SessionConfig(carry_over_slots=True, session_expiration_time=1 / 60),
    )

    now = time.time()

    # the tracker initially contains events
    initial_events = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=now - 10),
        SessionStarted(timestamp=now - 9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=now - 8),
        UserUttered(
            "/greet", {INTENT_NAME_KEY: "greet", "confidence": 1.0}, timestamp=now - 7
        ),
    ]

    tracker = DialogueStateTracker.from_events(conversation_id, initial_events)

    default_processor.tracker_store.save(tracker)

    tracker = await default_processor.fetch_tracker_with_initial_session(
        conversation_id, default_channel
    )

    # the conversation session has expired, but calling
    # `fetch_tracker_with_initial_session()` did not update it
    assert default_processor._has_session_expired(tracker)
    assert [event.as_dict() for event in tracker.events] == [
        event.as_dict() for event in initial_events
    ]


async def test_handle_message_with_session_start(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    model_id = default_processor.model_metadata.model_id

    entity = "name"
    slot_1 = {entity: "Core"}
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_1)}", default_channel, sender_id)
    )

    assert default_channel.latest_output() == {
        "recipient_id": sender_id,
        "text": "hey there Core!",
    }

    # patch processor so a session start is triggered
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    slot_2 = {entity: "post-session start hello"}
    # handle a new message
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_2)}", default_channel, sender_id)
    )

    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    # make sure the sequence of events is as expected
    expected = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_1)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 22,
                        "value": "Core",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_1[entity]),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted("utter_greet"),
            BotUttered(
                "hey there Core!",
                metadata={"utter_action": "utter_greet"},
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            # the initial SlotSet is reapplied after the SessionStarted sequence
            SlotSet(entity, slot_1[entity]),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_2)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 42,
                        "value": "post-session start hello",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_2[entity]),
            DefinePrevUserUtteredFeaturization(False),
            ActionExecuted("utter_greet"),
            BotUttered(
                "hey there post-session start hello!",
                metadata={"utter_action": "utter_greet"},
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    assert list(tracker.events) == expected


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


async def test_action_unlikely_intent_metadata(default_processor: MessageProcessor):
    tracker = DialogueStateTracker.from_events(
        "some-sender",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
    )
    domain = Domain.empty()
    metadata = {"key1": 1, "key2": "2"}

    await default_processor._run_action(
        ActionUnlikelyIntent(),
        tracker,
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator(domain.responses),
        PolicyPrediction([], "some policy", action_metadata=metadata),
    )

    applied_events = tracker.applied_events()
    assert applied_events == [
        ActionExecuted(ACTION_LISTEN_NAME),
        ActionExecuted(ACTION_UNLIKELY_INTENT_NAME, metadata=metadata),
    ]
    assert applied_events[1].metadata == metadata


async def test_restart_triggers_session_start(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
):
    sender_id = uuid.uuid4().hex
    model_id = default_processor.model_metadata.model_id

    entity = "name"
    slot_1 = {entity: "name1"}
    await default_processor.handle_message(
        UserMessage(f"/greet{json.dumps(slot_1)}", default_channel, sender_id)
    )

    assert default_channel.latest_output() == {
        "recipient_id": sender_id,
        "text": "hey there name1!",
    }

    # This restarts the chat
    await default_processor.handle_message(
        UserMessage("/restart", default_channel, sender_id)
    )

    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    expected = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                f"/greet{json.dumps(slot_1)}",
                {INTENT_NAME_KEY: "greet", "confidence": 1.0},
                [
                    {
                        "entity": entity,
                        "start": 6,
                        "end": 23,
                        "value": "name1",
                        "extractor": "RegexMessageHandler",
                    }
                ],
            ),
            SlotSet(entity, slot_1[entity]),
            DefinePrevUserUtteredFeaturization(use_text_for_featurization=False),
            ActionExecuted("utter_greet"),
            BotUttered(
                "hey there name1!",
                metadata={"utter_action": "utter_greet"},
            ),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("/restart", {INTENT_NAME_KEY: "restart", "confidence": 1.0}),
            DefinePrevUserUtteredFeaturization(use_text_for_featurization=False),
            ActionExecuted(ACTION_RESTART_NAME),
            Restarted(),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            # No previous slot is set due to restart.
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    for actual, expected in zip(tracker.events, expected):
        assert actual == expected


async def test_handle_message_if_action_manually_rejects(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    conversation_id = "test"
    message = UserMessage("/greet", sender_id=conversation_id)

    rejection_events = [
        SlotSet("my_slot", "test"),
        ActionExecutionRejected("utter_greet"),
        SlotSet("some slot", "some value"),
    ]

    async def mocked_run(self, *args: Any, **kwargs: Any) -> List[Event]:
        return rejection_events

    monkeypatch.setattr(ActionBotResponse, ActionBotResponse.run.__name__, mocked_run)
    await default_processor.handle_message(message)

    tracker = default_processor.tracker_store.retrieve(conversation_id)

    logged_events = list(tracker.events)

    assert ActionExecuted("utter_greet") not in logged_events
    assert all(event in logged_events for event in rejection_events)


async def test_policy_events_are_applied_to_tracker(
    default_processor: MessageProcessor, monkeypatch: MonkeyPatch
):
    model_id = default_processor.model_metadata.model_id
    expected_action = ACTION_LISTEN_NAME
    policy_events = [LoopInterrupted(True)]
    conversation_id = "test_policy_events_are_applied_to_tracker"
    user_message = "/greet"

    expected_events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(user_message, intent={"name": "greet"}),
            *policy_events,
        ],
        model_id,
    )

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        prediction = PolicyPrediction.for_action_name(
            default_processor.domain, expected_action, "some policy"
        )
        prediction.events = policy_events

        return prediction

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    action_received_events = False

    async def mocked_run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
    ) -> List[Event]:
        # The action already has access to the policy events
        nonlocal action_received_events
        action_received_events = list(tracker.events) == expected_events
        return []

    monkeypatch.setattr(ActionListen, ActionListen.run.__name__, mocked_run)

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    assert action_received_events

    tracker = default_processor.get_tracker(conversation_id)
    # The action was logged on the tracker as well
    expected_events.append(with_model_id(ActionExecuted(ACTION_LISTEN_NAME), model_id))

    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    "reject_fn",
    [
        lambda: [ActionExecutionRejected(ACTION_LISTEN_NAME)],
        lambda: (_ for _ in ()).throw(ActionExecutionRejection(ACTION_LISTEN_NAME)),
    ],
)
async def test_policy_events_not_applied_if_rejected(
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
    reject_fn: Callable[[], List[Event]],
):
    model_id = default_processor.model_metadata.model_id
    expected_action = ACTION_LISTEN_NAME
    expected_events = [LoopInterrupted(True)]
    conversation_id = "test_policy_events_are_applied_to_tracker"
    user_message = "/greet"

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        prediction = PolicyPrediction.for_action_name(
            default_processor.domain, expected_action, "some policy"
        )
        prediction.events = expected_events

        return prediction

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    async def mocked_run(*args: Any, **kwargs: Any) -> List[Event]:
        return reject_fn()

    monkeypatch.setattr(ActionListen, ActionListen.run.__name__, mocked_run)

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    tracker = default_processor.get_tracker(conversation_id)
    expected_events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                user_message,
                intent={"name": "greet"},
            ),
            ActionExecutionRejected(ACTION_LISTEN_NAME),
        ],
        model_id,
    )
    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


async def test_logging_of_end_to_end_action(
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    model_id = default_processor.model_metadata.model_id
    end_to_end_action = "hi, how are you?"
    new_domain = Domain(
        intents=["greet"],
        entities=[],
        slots=[],
        responses={},
        action_names=[],
        forms={},
        action_texts=[end_to_end_action],
    )

    default_processor.domain = new_domain

    conversation_id = "test_logging_of_end_to_end_action"
    user_message = "/greet"

    number_of_calls = 0

    def combine_predictions(
        self,
        predictions: List[PolicyPrediction],
        tracker: DialogueStateTracker,
        domain: Domain,
        **kwargs: Any,
    ) -> PolicyPrediction:
        nonlocal number_of_calls
        if number_of_calls == 0:
            prediction = PolicyPrediction.for_action_name(
                new_domain, end_to_end_action, "some policy"
            )
            prediction.is_end_to_end_prediction = True
            number_of_calls += 1
            return prediction
        else:
            return PolicyPrediction.for_action_name(new_domain, ACTION_LISTEN_NAME)

    monkeypatch.setattr(
        DefaultPolicyPredictionEnsemble, "combine_predictions", combine_predictions
    )

    await default_processor.handle_message(
        UserMessage(user_message, sender_id=conversation_id)
    )

    tracker = default_processor.tracker_store.retrieve(conversation_id)
    expected_events = with_model_ids(
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(
                user_message,
                intent={"name": "greet"},
            ),
            ActionExecuted(action_text=end_to_end_action),
            BotUttered("hi, how are you?", {}, {}, 123),
            ActionExecuted(ACTION_LISTEN_NAME),
        ],
        model_id=model_id,
    )
    for event, expected in zip(tracker.events, expected_events):
        assert event == expected


async def test_predict_next_action_with_hidden_rules(
    trained_async: Callable, tmp_path: Path
):
    rule_intent = "rule_intent"
    rule_action = "rule_action"
    story_intent = "story_intent"
    story_action = "story_action"
    rule_slot = "rule_slot"
    story_slot = "story_slot"
    domain_content = textwrap.dedent(
        f"""
        version: "2.0"
        intents:
        - {rule_intent}
        - {story_intent}
        actions:
        - {rule_action}
        - {story_action}
        slots:
          {rule_slot}:
            type: text
            mappings:
            - type: from_text
          {story_slot}:
            type: text
            mappings:
            - type: from_text
        """
    )
    domain = Domain.from_yaml(domain_content)
    domain_path = tmp_path / "domain.yml"
    rasa.shared.utils.io.write_text_file(domain_content, domain_path)

    training_data = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"

    rules:
    - rule: rule
      steps:
      - intent: {rule_intent}
      - action: {rule_action}
      - slot_was_set:
          - {rule_slot}: {rule_slot}

    stories:
    - story: story
      steps:
      - intent: {story_intent}
      - action: {story_action}
      - slot_was_set:
          - {story_slot}: {story_slot}
    """
    )
    training_data_path = tmp_path / "data.yml"
    rasa.shared.utils.io.write_text_file(training_data, training_data_path)

    config = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    policies:
    - name: RulePolicy
    - name: MemoizationPolicy

    """
    )
    config_path = tmp_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)
    model_path = await trained_async(
        str(domain_path), str(config_path), [str(training_data_path)]
    )
    agent = await load_agent(model_path=model_path)
    processor = agent.processor

    tracker = DialogueStateTracker.from_events(
        "casd",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered(intent={"name": rule_intent}),
        ],
        slots=domain.slots,
    )
    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert action._name == rule_action
    assert prediction.hide_rule_turn

    processor._log_action_on_tracker(
        tracker, action, [SlotSet(rule_slot, rule_slot)], prediction
    )

    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert isinstance(action, ActionListen)
    assert prediction.hide_rule_turn

    processor._log_action_on_tracker(tracker, action, None, prediction)

    tracker.events.append(UserUttered(intent={"name": story_intent}))

    # rules are hidden correctly if memo policy predicts next actions correctly
    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert action._name == story_action
    assert not prediction.hide_rule_turn

    processor._log_action_on_tracker(
        tracker, action, [SlotSet(story_slot, story_slot)], prediction
    )

    action, prediction = processor.predict_next_with_tracker_if_should(tracker)
    assert isinstance(action, ActionListen)
    assert not prediction.hide_rule_turn


def test_predict_next_action_raises_limit_reached_exception(
    default_processor: MessageProcessor,
):
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            ActionExecuted(ACTION_LISTEN_NAME),
            UserUttered("Hi!"),
            ActionExecuted("test_action"),
        ],
    )
    tracker.set_latest_action({"action_name": "test_action"})

    default_processor.max_number_of_predictions = 1
    with pytest.raises(ActionLimitReached):
        default_processor.predict_next_with_tracker_if_should(tracker)


async def test_processor_logs_text_tokens_in_tracker(
    mood_agent: Agent, whitespace_tokenizer: WhitespaceTokenizer
):
    text = "Hello there"
    tokens = whitespace_tokenizer.tokenize(Message(data={"text": text}), "text")
    indices = [(t.start, t.end) for t in tokens]

    message = UserMessage(text)
    processor = mood_agent.processor
    tracker = await processor.log_message(message)
    event = tracker.get_last_event_for(event_type=UserUttered)
    event_tokens = event.as_dict().get("parse_data").get("text_tokens")

    assert event_tokens == indices


async def test_processor_valid_slot_setting(form_bot_agent: Agent):
    processor = form_bot_agent.processor
    message = UserMessage(
        "that's correct",
        CollectingOutputChannel(),
        "test",
        parse_data={
            "intent": {"name": "affirm"},
            "entities": [{"entity": "seating", "value": True}],
        },
    )
    await processor.handle_message(message)
    tracker = processor.get_tracker("test")
    assert SlotSet("outdoor_seating", True) in tracker.events


async def test_parse_message_nlu_only(trained_moodbot_nlu_path: Text):
    processor = Agent.load(model_path=trained_moodbot_nlu_path).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert result["intent"]["name"]


async def test_parse_message_core_only(trained_core_model: Text):
    processor = Agent.load(model_path=trained_core_model).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert not result["intent"]["name"]


async def test_parse_message_full_model(trained_moodbot_path: Text):
    processor = Agent.load(model_path=trained_moodbot_path).processor
    message = UserMessage("/greet")
    result = await processor.parse_message(message)
    assert result == {
        "text": "/greet",
        "intent": {"name": "greet", "confidence": 1.0},
        "intent_ranking": [{"name": "greet", "confidence": 1.0}],
        "entities": [],
    }

    message = UserMessage("Hello")
    result = await processor.parse_message(message)
    assert result["intent"]["name"]


def test_predict_next_with_tracker_nlu_only(trained_nlu_model: Text):
    processor = Agent.load(model_path=trained_nlu_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result is None


def test_predict_next_with_tracker_core_only(trained_core_model: Text):
    processor = Agent.load(model_path=trained_core_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result["policy"] == "MemoizationPolicy"


def test_predict_next_with_tracker_full_model(trained_rasa_model: Text):
    processor = Agent.load(model_path=trained_rasa_model).processor
    tracker = DialogueStateTracker("some_id", [])
    tracker.followup_action = None
    result = processor.predict_next_with_tracker(tracker)
    assert result["policy"] == "MemoizationPolicy"


def test_get_tracker_adds_model_id(default_processor: MessageProcessor):
    model_id = default_processor.model_metadata.model_id
    tracker = default_processor.get_tracker("bloop")
    assert tracker.model_id == model_id


async def test_processor_e2e_slot_set(e2e_bot_agent: Agent, caplog: LogCaptureFixture):
    processor = e2e_bot_agent.processor
    message = UserMessage("I am feeling sad.", CollectingOutputChannel(), "test",)
    with caplog.at_level(logging.DEBUG):
        await processor.handle_message(message)

    tracker = processor.get_tracker("test")
    assert SlotSet("mood", "sad") in tracker.events
    assert any(
        "An end-to-end prediction was made which has triggered the 2nd execution of "
        "the default action 'action_extract_slots'." in message
        for message in caplog.messages
    )
