import uuid
from typing import List
from unittest.mock import Mock

import pytest
from pymongo.errors import OperationFailure
from pytest import MonkeyPatch

from rasa.core.agent import Agent
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    SessionStarted,
    UserUttered,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ConnectionException
from rasa.utils.endpoints import EndpointConfig
from tests.core.conftest import MockedMongoTrackerStore
from tests.core.tracker_stores.conftest import (
    _saved_tracker_with_multiple_session_starts,
    create_tracker_with_partially_saved_events,
    prepare_token_serialisation,
)


def test_mongo_tracker_store_raise_exception(domain: Domain, monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        "rasa.core.tracker_stores.mongo_tracker_store.MongoTrackerStore",
        Mock(
            side_effect=OperationFailure("not authorized on logs to execute command.")
        ),
    )
    with pytest.raises(ConnectionException) as error:
        TrackerStore.create(
            EndpointConfig(username="username", password="password", type="mongod"),
            domain,
        )

    assert "not authorized on logs to execute command." in str(error.value)


async def test_mongo_additional_events(domain: Domain):
    tracker_store = MockedMongoTrackerStore(domain)
    events, tracker = await create_tracker_with_partially_saved_events(tracker_store)

    # make sure only new events are returned
    # noinspection PyProtectedMember
    assert list(tracker_store._additional_events(tracker)) == events


async def test_mongo_additional_events_with_session_start(domain: Domain):
    sender = "test_mongo_additional_events_with_session_start"
    tracker_store = MockedMongoTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(tracker_store, sender)

    tracker.update(UserUttered("hi2"))

    # noinspection PyProtectedMember
    additional_events = list(tracker_store._additional_events(tracker))

    assert len(additional_events) == 1
    assert isinstance(additional_events[0], UserUttered)


def test_current_state_without_events(domain: Domain):
    tracker_store = MockedMongoTrackerStore(domain)

    # insert some events
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_mongo_tracker_store_current_state_without_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)

    # get current state without events
    # noinspection PyProtectedMember
    state = tracker_store._current_tracker_state_without_events(tracker)

    # `events` key should not be in there
    assert state and "events" not in state


def test_mongo_tracker_store_with_token_serialisation(
    domain: Domain, response_selector_agent: Agent
):
    tracker_store = MockedMongoTrackerStore(domain)
    prepare_token_serialisation(tracker_store, response_selector_agent, "mongo")


async def test_mongo_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_mongo_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)

    # the retrieved tracker with the latest session would not contain
    # `action_session_start` event because the MongoTrackerStore filters
    # only the events after `session_started` event
    assert list(tracker.events) == events_after_restart[1:]


def test_mongo_tracker_store_connection_error(domain: Domain):
    store = EndpointConfig.from_dict(
        {
            "type": "mongod",
            "url": "mongodb://0.0.0.0:42/?serverSelectionTimeoutMS=5000",
        }
    )

    with pytest.raises(ConnectionException):
        TrackerStore.create(store, domain)


async def test_tracker_store_counts_conversations() -> None:
    tracker_store = MockedMongoTrackerStore(Domain.empty())

    # Create two trackers
    tracker1 = DialogueStateTracker.from_events("1", [SessionStarted(timestamp=1)])
    tracker2 = DialogueStateTracker.from_events("2", [SessionStarted(timestamp=3)])
    await tracker_store.save(tracker1)
    await tracker_store.save(tracker2)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 2
    assert await tracker_store.count_conversations(after_timestamp=2) == 1
    assert await tracker_store.count_conversations(after_timestamp=4) == 0

    # Create another tracker
    tracker3 = DialogueStateTracker.from_events("3", [SessionStarted(timestamp=5)])
    await tracker_store.save(tracker3)

    # Assert that the tracker store counts the conversations correctly
    assert await tracker_store.count_conversations() == 3
    assert await tracker_store.count_conversations(after_timestamp=4) == 1


async def test_tracker_store_retrieve_with_session_started_events_mongo(
    domain: Domain,
):
    tracker_store = MockedMongoTrackerStore(domain)
    events = [
        UserUttered("Hola", {"name": "greet"}, timestamp=1),
        BotUttered("Hi", timestamp=2),
        SessionStarted(timestamp=3),
        UserUttered("Ciao", {"name": "greet"}, timestamp=4),
    ]
    sender_id = "test_sql_tracker_store_with_session_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Retrieve tracker with events since latest SessionStarted
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 2
    assert all((event == tracker.events[i] for i, event in enumerate(events[2:])))


async def test_tracker_store_retrieve_without_session_started_events_mongo(
    domain,
) -> None:
    tracker_store = MockedMongoTrackerStore(domain)

    # Create tracker with a SessionStarted event
    events = [
        UserUttered("Hola", {"name": "greet"}),
        BotUttered("Hi"),
        UserUttered("Ciao", {"name": "greet"}),
        BotUttered("Hi2"),
    ]

    sender_id = "test_sql_tracker_store_retrieve_without_session_started_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 4
    assert all(event == tracker.events[i] for i, event in enumerate(events))


async def test_mongo_tracker_store_retrieve_with_events_from_previous_sessions() -> (
    None
):
    tracker_store = MockedMongoTrackerStore(Domain.empty())

    conversation_id = uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            UserUttered("hi"),
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
        ],
    )
    await tracker_store.save(tracker)

    actual = await tracker_store.retrieve_full_tracker(conversation_id)

    assert len(actual.events) == len(tracker.events)
