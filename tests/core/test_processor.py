import asyncio

import datetime
import pytest
import time
import uuid
import json
from _pytest.monkeypatch import MonkeyPatch
from aioresponses import aioresponses
from typing import Optional, Text, List, Callable
from unittest.mock import patch, Mock

from rasa.core import jobs
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME

from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from rasa.core.domain import SessionConfig, Domain
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
from rasa.core.interpreter import RasaNLUHttpInterpreter, NaturalLanguageInterpreter
from rasa.core.policies import SimplePolicyEnsemble
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.processor import MessageProcessor
from rasa.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore
from rasa.core.trackers import DialogueStateTracker
from rasa.nlu.constants import INTENT_NAME_KEY
from rasa.utils.endpoints import EndpointConfig
from tests.utilities import latest_request

from rasa.core.constants import EXTERNAL_MESSAGE_PREFIX, IS_EXTERNAL, DEFAULT_INTENTS

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
    message = UserMessage('/dislike{"test_entity": "RASA"}')
    parsed = await default_processor.parse_message(message)
    with pytest.warns(UserWarning) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 2

    assert (
        record[0].message.args[0].startswith("Interpreter parsed an intent 'dislike'")
    )
    assert (
        record[1]
        .message.args[0]
        .startswith("Interpreter parsed an entity 'test_entity'")
    )


@pytest.mark.parametrize("default_intent", DEFAULT_INTENTS)
async def test_default_intent_recognized(
    default_processor: MessageProcessor, default_intent: Text
):
    message = UserMessage(default_intent)
    parsed = await default_processor.parse_message(message)
    with pytest.warns(None) as record:
        default_processor._check_for_unseen_features(parsed)
    assert len(record) == 0


async def test_http_parsing():
    message = UserMessage("lunch?")

    endpoint = EndpointConfig("https://interpreter.com")
    with aioresponses() as mocked:
        mocked.post("https://interpreter.com/model/parse", repeat=True, status=200)

        inter = RasaNLUHttpInterpreter(endpoint_config=endpoint)
        try:
            await MessageProcessor(inter, None, None, None, None).parse_message(message)
        except KeyError:
            pass  # logger looks for intent and entities, so we except

        r = latest_request(mocked, "POST", "https://interpreter.com/model/parse")

        assert r


async def mocked_parse(self, text, message_id=None, tracker=None):
    """Mock parsing a text message and augment it with the slot
    value from the tracker's state."""

    return {
        "intent": {INTENT_NAME_KEY: "", "confidence": 0.0},
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

    reminder = ReminderScheduled("remind", datetime.datetime.now())
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    tracker.update(UserUttered("test"))
    tracker.update(ActionExecuted("action_schedule_reminder"))
    tracker.update(reminder)

    default_processor.tracker_store.save(tracker)

    await default_processor.handle_reminder(
        reminder, sender_id, default_channel, default_processor.nlg
    )

    # retrieve the updated tracker
    t = default_processor.tracker_store.retrieve(sender_id)

    assert t.events[-5] == UserUttered("test")
    assert t.events[-4] == ActionExecuted("action_schedule_reminder")
    assert isinstance(t.events[-3], ReminderScheduled)
    assert t.events[-2] == UserUttered(
        f"{EXTERNAL_MESSAGE_PREFIX}remind",
        intent={INTENT_NAME_KEY: "remind", IS_EXTERNAL: True},
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
            tracker.events, tracker, default_channel, default_processor.nlg
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
        tracker.events, tracker, default_channel, default_processor.nlg
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
        tracker.events, tracker, default_channel, default_processor.nlg
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
        tracker.events, tracker, default_channel, default_processor.nlg
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
        tracker.events, tracker, default_channel, default_processor.nlg
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


# noinspection PyProtectedMember
async def test_update_tracker_session_with_metadata(
    default_channel: CollectingOutputChannel,
    default_processor: MessageProcessor,
    monkeypatch: MonkeyPatch,
):
    sender_id = uuid.uuid4().hex
    tracker = default_processor.tracker_store.get_or_create_tracker(sender_id)

    # patch `_has_session_expired()` so the `_update_tracker_session()` call actually
    # does something
    monkeypatch.setattr(default_processor, "_has_session_expired", lambda _: True)

    metadata = {"metadataTestKey": "metadataTestValue"}

    await default_processor._update_tracker_session(tracker, default_channel, metadata)

    # the save is not called in _update_tracker_session()
    default_processor._save_tracker(tracker)

    # inspect tracker events and make sure SessionStarted event is present
    # and has metadata.
    tracker = default_processor.tracker_store.retrieve(sender_id)
    assert tracker.events.count(SessionStarted()) == 1

    session_started_event_idx = tracker.events.index(SessionStarted())
    session_started_event_metadata = tracker.events[session_started_event_idx].metadata

    assert session_started_event_metadata == metadata


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
    assert events[7:9] == [ActionExecuted(ACTION_SESSION_START_NAME), SessionStarted()]

    # the five slots should be reapplied
    assert events[9:14] == slot_set_events

    # finally an action listen, this should also be the last event
    assert events[14] == events[-1] == ActionExecuted(ACTION_LISTEN_NAME)


# noinspection PyProtectedMember
async def test_get_tracker_with_session_start(
    default_channel: CollectingOutputChannel, default_processor: MessageProcessor
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
    assert list(tracker.events) == [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
        UserUttered(
            f"/greet{json.dumps(slot_1)}",
            {INTENT_NAME_KEY: "greet", "confidence": 1.0},
            [{"entity": entity, "start": 6, "end": 22, "value": "Core"}],
        ),
        SlotSet(entity, slot_1[entity]),
        ActionExecuted("utter_greet"),
        BotUttered("hey there Core!", metadata={"template_name": "utter_greet"}),
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


def test_get_next_action_probabilities_passes_interpreter_to_policies(
    monkeypatch: MonkeyPatch,
):
    policy = TEDPolicy()
    test_interpreter = Mock()

    def predict_action_probabilities(
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs,
    ) -> List[float]:
        assert interpreter == test_interpreter
        return [1, 0]

    policy.predict_action_probabilities = predict_action_probabilities
    ensemble = SimplePolicyEnsemble(policies=[policy])

    domain = Domain.empty()

    processor = MessageProcessor(
        test_interpreter, ensemble, domain, InMemoryTrackerStore(domain), Mock()
    )

    # This should not raise
    processor._get_next_action_probabilities(
        DialogueStateTracker.from_events("lala", [ActionExecuted(ACTION_LISTEN_NAME)])
    )


@pytest.mark.parametrize(
    "predict_function",
    [lambda tracker, domain: [1, 0], lambda tracker, domain, some_bool=True: [1, 0]],
)
def test_get_next_action_probabilities_pass_policy_predictions_without_interpreter_arg(
    predict_function: Callable,
):
    policy = TEDPolicy()

    policy.predict_action_probabilities = predict_function

    ensemble = SimplePolicyEnsemble(policies=[policy])
    interpreter = Mock()
    domain = Domain.empty()

    processor = MessageProcessor(
        interpreter, ensemble, domain, InMemoryTrackerStore(domain), Mock()
    )

    with pytest.warns(DeprecationWarning):
        processor._get_next_action_probabilities(
            DialogueStateTracker.from_events(
                "lala", [ActionExecuted(ACTION_LISTEN_NAME)]
            )
        )
