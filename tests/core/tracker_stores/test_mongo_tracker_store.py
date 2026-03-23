import uuid
from typing import List, Optional, Text
from unittest.mock import Mock

import pytest
from pymongo.errors import OperationFailure
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.agent import Agent
from rasa.core.tracker_stores.tracker_store import TrackerStore
from rasa.shared.constants import DEFAULT_SENDER_ID, DEFAULT_USER_ID
from rasa.shared.core.constants import ACTION_SESSION_START_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    ConversationInactive,
    DialogueStackUpdated,
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
    assert_all_trackers_have_user_id,
    assert_tracker_has_user_id,
    create_multiple_trackers_with_user_id,
    create_tracker_with_partially_saved_events,
    create_tracker_with_user_id,
    create_trackers_with_same_timestamp,
    old_tracker_gets_timestamp_on_save,
    old_tracker_gets_timestamp_on_update,
    prepare_token_serialisation,
)
from tests.utilities import filter_logs


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


async def test_mongo_additional_events_domain_none():
    """When the domain is empty, _additional_events returns an empty iterator (no silent loss)."""
    domain = Domain.empty()
    tracker_store = MockedMongoTrackerStore(domain)

    tracker = DialogueStateTracker.from_events(
        "sender_domain_none", [UserUttered("hello")]
    )
    await tracker_store.save(tracker)
    # noinspection PyProtectedMember
    result = list(tracker_store._additional_events(tracker))

    assert result == []


def test_events_since_last_action_session_start_no_match_returns_all():
    """When no action_session_start action exists, all events are returned."""
    from rasa.core.tracker_stores.mongo_tracker_store import MongoTrackerStore

    events = [
        {"event": "user", "text": "hello"},
        {"event": "bot", "text": "hi"},
        {"event": "session_started"},
    ]
    # noinspection PyProtectedMember
    result = MongoTrackerStore._events_since_last_action_session_start(events)

    assert result == events


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


@pytest.mark.asyncio
async def test_mongo_tracker_store_with_token_serialisation(
    domain: Domain, response_selector_agent: Agent
):
    tracker_store = MockedMongoTrackerStore(domain)
    await prepare_token_serialisation(tracker_store, response_selector_agent, "mongo")


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

    # Latest session begins at the last ``action_session_start`` (inclusive).
    assert list(tracker.events) == events_after_restart


async def test_mongo_tracker_store_retrieve_latest_session_with_stack_events() -> None:
    """Same stack scenario as SQL: align with SQL fixture (see SQL test docstring).

    Do not emit ``SessionStarted`` between the second ``action_session_start`` and
    the stack ``replace``: ``SessionStarted`` resets the tracker (including the
    stack), so a replace on ``/0`` would fail during replay.
    """
    tracker_store = MockedMongoTrackerStore(Domain.empty())
    sender_id = "mongo_stack_latest"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            SessionStarted(),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "old", "flow_id": "foo", "step_id": "OLD", "frame_type": "regular", "type": "flow"}}]'
            ),
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "SECOND"}]'
            ),
        ],
    )
    await tracker_store.save(tracker)

    retrieved = await tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert next(iter(retrieved.events)).type_name == ActionExecuted.type_name
    assert next(iter(retrieved.events)).action_name == ACTION_SESSION_START_NAME
    assert retrieved.stack.frames[0].frame_id == "old"
    assert retrieved.stack.frames[0].step_id == "SECOND"


async def test_mongo_tracker_store_retrieve_widens_prefix_for_stack_integrity_true_mode() -> (
    None
):
    """Replay-safe widening across action_session_start when start_session_after_expiry is True."""
    domain = Domain.from_dict({"session_config": {"start_session_after_expiry": True}})
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = "mongo_widen_true"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ActionExecuted(ACTION_SESSION_START_NAME),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "SECOND"}]'
            ),
        ],
    )
    await tracker_store.save(tracker)

    retrieved = await tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 4
    assert retrieved.stack.frames[0].step_id == "SECOND"


async def test_mongo_tracker_store_retrieve_widens_prefix_for_stack_integrity_false_mode() -> (
    None
):
    """Boundary after ConversationInactive when start_session_after_expiry is False."""
    domain = Domain.from_dict({"session_config": {"start_session_after_expiry": False}})
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = "mongo_widen_false"
    tracker = DialogueStateTracker.from_events(
        sender_id,
        [
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "f1", "flow_id": "foo", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'
            ),
            ConversationInactive(),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "AFTER_INACTIVE"}]'
            ),
        ],
    )
    await tracker_store.save(tracker)

    retrieved = await tracker_store.retrieve(sender_id)

    assert retrieved is not None
    assert len(retrieved.events) == 3
    assert retrieved.stack.frames[0].step_id == "AFTER_INACTIVE"


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
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=2.5),
        SessionStarted(timestamp=3),
        UserUttered("Ciao", {"name": "greet"}, timestamp=4),
    ]
    sender_id = "test_sql_tracker_store_with_session_events"
    tracker = DialogueStateTracker.from_events(sender_id, events)
    await tracker_store.save(tracker)

    # Save other tracker to ensure that we don't run into problems with other senders
    other_tracker = DialogueStateTracker.from_events("other-sender", [SessionStarted()])
    await tracker_store.save(other_tracker)

    # Latest session begins at ``action_session_start`` (inclusive).
    tracker = await tracker_store.retrieve(sender_id)

    assert len(tracker.events) == 3
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


async def test_mongo_tracker_store_delete_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    # Given
    tracker_store = MockedMongoTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    # When
    with capture_logs() as caplog:
        await tracker_store.delete(sender_id)
        logs = filter_logs(
            caplog,
            event="mongo_tracker_store.delete.deleted_tracker",
            log_level="info",
        )

        assert len(logs) == 1

    # Then
    retrieved_tracker = await tracker_store.retrieve(sender_id)
    assert retrieved_tracker is None


async def test_mongo_tracker_store_delete_no_tracker(
    domain: Domain,
) -> None:
    with capture_logs() as caplog:
        tracker_store = MockedMongoTrackerStore(domain)
        sender_id = uuid.uuid4().hex
        await tracker_store.delete(sender_id)
        logs = filter_logs(
            caplog,
            event="mongo_tracker_store.delete.no_tracker_for_sender_id",
            log_level="info",
            log_message_parts=[
                f"Could not find tracker for conversation ID '{sender_id}'."
            ],
        )

        assert len(logs) == 1


async def test_mongo_tracker_store_update_tracker(domain: Domain) -> None:
    # Given
    sender_id = uuid.uuid4().hex
    tracker_store = MockedMongoTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(
        tracker_store, sender_id
    )
    new_events = list(tracker.events)[3:] + [
        UserUttered("What's the weather like today?")
    ]
    new_tracker = DialogueStateTracker.from_events(
        sender_id,
        new_events,
    )

    # When
    await tracker_store.update(new_tracker)

    # Then
    updated_tracker = await tracker_store.retrieve(sender_id)
    assert updated_tracker == new_tracker


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_mongo_tracker_store_preserves_user_id(
    domain: Domain, user_id: Optional[Text]
) -> None:
    """Test MongoTrackerStore preserves user_id on save/retrieve."""
    # Given
    conversation_id = uuid.uuid4().hex
    tracker_store = MockedMongoTrackerStore(domain)

    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [SessionStarted(), UserUttered("Hello")],
        user_id=user_id,
    )

    # When
    await tracker_store.save(tracker)

    # Then
    retrieved_tracker = await tracker_store.retrieve(conversation_id)
    assert retrieved_tracker is not None
    assert retrieved_tracker.user_id == user_id

    # Test with retrieve_full_tracker as well
    retrieved_full_tracker = await tracker_store.retrieve_full_tracker(conversation_id)
    assert retrieved_full_tracker is not None
    assert retrieved_full_tracker.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_mongo_tracker_store_get_or_create(
    domain: Domain, user_id: Optional[Text]
) -> None:
    """Test get_or_create_tracker with and without user_id."""
    # Given
    tracker_store = MockedMongoTrackerStore(domain)

    # When
    tracker = await tracker_store.get_or_create_tracker(
        DEFAULT_SENDER_ID, user_id=user_id
    )

    # Then
    assert tracker.user_id == user_id

    retrieved = await tracker_store.get_or_create_tracker(DEFAULT_SENDER_ID)
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_mongo_tracker_store_update_preserves_user_id(
    domain: Domain, user_id: Optional[Text]
) -> None:
    """Test update operation preserves user_id."""
    # Given
    tracker_store = MockedMongoTrackerStore(domain)

    # Create and save initial tracker
    tracker = DialogueStateTracker.from_events(
        DEFAULT_SENDER_ID,
        [SessionStarted(), UserUttered("Hello")],
        user_id=user_id,
    )
    await tracker_store.save(tracker)

    # When
    await tracker_store.update(tracker)

    # Then
    retrieved = await tracker_store.retrieve(DEFAULT_SENDER_ID)

    assert retrieved is not None
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [DEFAULT_USER_ID, None])
async def test_mongo_tracker_store_save_appends_events_preserves_user_id(
    domain: Domain, user_id: Optional[Text]
) -> None:
    """Test incremental saves preserve user_id."""
    # Given
    tracker_store = MockedMongoTrackerStore(domain)

    # Create initial tracker
    tracker = DialogueStateTracker.from_events(
        DEFAULT_SENDER_ID,
        [SessionStarted(), UserUttered("Hello")],
        user_id=user_id,
    )
    await tracker_store.save(tracker)

    # When
    tracker.update_with_events(
        [ActionExecuted("action_listen"), BotUttered("Hi")],
        domain=domain,
    )
    await tracker_store.save(tracker)

    # Then
    retrieved = await tracker_store.retrieve(DEFAULT_SENDER_ID)
    assert retrieved is not None
    assert retrieved.user_id == user_id


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id(domain: Domain) -> None:
    """Test get_trackers_by_user_id filters by user_id correctly."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create trackers with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)
    await create_tracker_with_user_id(
        tracker_store, "sender2", user_id, [SessionStarted(), UserUttered("hi")]
    )

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store, "sender3", "user_456", [SessionStarted(), UserUttered("hey")]
    )

    # Create tracker without user_id
    await create_tracker_with_user_id(
        tracker_store, "sender4", None, [SessionStarted(), UserUttered("ho")]
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 2
    sender_ids = {tracker.sender_id for tracker in trackers}
    assert "sender1" in sender_ids
    assert "sender2" in sender_ids
    assert "sender3" not in sender_ids
    assert "sender4" not in sender_ids

    # Verify all trackers have correct user_id
    for tracker in trackers:
        assert tracker.user_id == user_id


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_no_matches(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id returns empty list when no trackers match."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create tracker with different user_id
    await create_tracker_with_user_id(tracker_store, "sender1", "user_456")

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 0


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_filters_no_user_id(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id filters out trackers without user_id."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create tracker with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)

    # Create tracker without user_id
    await create_tracker_with_user_id(
        tracker_store, "sender2", None, [SessionStarted(), UserUttered("hi")]
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], "sender1", user_id)


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_save_sets_user_id(
    domain: Domain,
) -> None:
    """Test that save method stores user_id in MongoDB."""
    # Given
    user_id = DEFAULT_USER_ID
    conversation_id = uuid.uuid4().hex
    tracker_store = MockedMongoTrackerStore(domain)

    tracker = await create_tracker_with_user_id(tracker_store, conversation_id, user_id)
    await tracker_store.save(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], conversation_id, user_id)


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_update_sets_user_id(
    domain: Domain,
) -> None:
    """Test that update method stores user_id in MongoDB."""
    # Given
    user_id = DEFAULT_USER_ID
    conversation_id = uuid.uuid4().hex
    tracker_store = MockedMongoTrackerStore(domain)

    # Create tracker with user_id
    tracker = DialogueStateTracker.from_events(
        conversation_id,
        [SessionStarted(), UserUttered("hello")],
        slots=domain.slots,
        user_id=user_id,
    )
    await tracker_store.update(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], conversation_id, user_id)


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_multiple_trackers(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id handles multiple trackers correctly."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create multiple trackers with the same user_id
    conversation_ids = []
    for i in range(50):
        conversation_id = uuid.uuid4().hex
        conversation_ids.append(conversation_id)
        await create_tracker_with_user_id(
            tracker_store,
            conversation_id,
            user_id,
            [SessionStarted(), UserUttered(f"Message {i}")],
        )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 50
    retrieved_ids = {tracker.sender_id for tracker in trackers}
    assert retrieved_ids == set(conversation_ids)

    # Verify all trackers have correct user_id
    for tracker in trackers:
        assert tracker.user_id == user_id


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_with_limit(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id respects limit parameter."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create multiple trackers with the same user_id
    for i in range(10):
        conversation_id = uuid.uuid4().hex
        tracker = DialogueStateTracker.from_events(
            conversation_id,
            [SessionStarted(), UserUttered(f"Message {i}")],
            slots=domain.slots,
            user_id=user_id,
        )
        await tracker_store.save(tracker)

    # When - request only 5 trackers
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=5)

    # Then
    assert len(trackers) == 5
    for tracker in trackers:
        assert tracker.user_id == user_id


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_with_skip(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id respects skip parameter."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)
    saved_trackers = []

    # Create multiple trackers with the same user_id
    conversation_ids = []
    for i in range(10):
        conversation_id = uuid.uuid4().hex
        conversation_ids.append(conversation_id)
        tracker = await create_tracker_with_user_id(
            tracker_store,
            conversation_id,
            user_id,
            [SessionStarted(), UserUttered(f"Message {i}")],
        )
        saved_trackers.append(tracker)

    # When - skip first 3 trackers
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Then
    assert len(trackers) == 7  # 10 total - 3 skipped
    assert_all_trackers_have_user_id(trackers, user_id)
    assert trackers == saved_trackers[3:]


@pytest.mark.asyncio
async def test_mongo_tracker_store_get_trackers_by_user_id_with_limit_and_skip(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id respects both limit and skip parameters."""
    # Given
    user_id = DEFAULT_USER_ID
    tracker_store = MockedMongoTrackerStore(domain)

    # Create multiple trackers with the same user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10, domain=domain
    )

    # When - skip first 2, then return next 3
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

    # Then
    assert len(trackers) == 3
    assert trackers == saved_trackers[2:5]
    assert_all_trackers_have_user_id(trackers, user_id)


# Backward compatibility tests for conversation_started_timestamp
@pytest.mark.asyncio
async def test_mongo_old_tracker_gets_timestamp_on_save(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it on save."""
    tracker_store = MockedMongoTrackerStore(domain)
    await old_tracker_gets_timestamp_on_save(tracker_store, domain=domain)


@pytest.mark.asyncio
async def test_mongo_old_tracker_gets_timestamp_on_update(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it
    on update."""
    tracker_store = MockedMongoTrackerStore(domain)
    await old_tracker_gets_timestamp_on_update(tracker_store, domain=domain)


# Sorting consistency tests
@pytest.mark.asyncio
async def test_mongo_sorting_by_sender_id_when_timestamps_identical(
    domain: Domain,
) -> None:
    """Test that trackers with identical timestamps are sorted by sender_id."""
    tracker_store = MockedMongoTrackerStore(domain)
    user_id = DEFAULT_USER_ID
    timestamp = 1234567890.0
    sender_ids = ["sender_c", "sender_a", "sender_b"]

    # Create trackers with same timestamp
    await create_trackers_with_same_timestamp(
        tracker_store, user_id, timestamp, sender_ids, domain=domain
    )

    # Retrieve and verify sorting
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Should be sorted by sender_id when timestamps are identical
    assert len(trackers) == 3
    assert trackers[0].sender_id == "sender_a"
    assert trackers[1].sender_id == "sender_b"
    assert trackers[2].sender_id == "sender_c"

    # All should have same timestamp
    for tracker in trackers:
        assert tracker.conversation_started_timestamp == timestamp


@pytest.mark.asyncio
async def test_mongo_pagination_very_large_skip(domain: Domain) -> None:
    """Test that very large skip values are handled gracefully."""
    tracker_store = MockedMongoTrackerStore(domain)
    user_id = DEFAULT_USER_ID

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large skip should return empty list
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=1000000)

    assert len(trackers) == 0


@pytest.mark.asyncio
async def test_mongo_pagination_very_large_limit(domain: Domain) -> None:
    """Test that very large limit values are handled gracefully."""
    tracker_store = MockedMongoTrackerStore(domain)
    user_id = DEFAULT_USER_ID

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large limit should return all items (up to available)
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=1000000)

    assert len(trackers) == 5


@pytest.mark.asyncio
async def test_mongo_negative_skip_and_limit_ignored(domain: Domain) -> None:
    """Test that both negative skip and limit values are ignored."""
    tracker_store = MockedMongoTrackerStore(domain)
    user_id = DEFAULT_USER_ID

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # When: Retrieve with both negative skip and limit
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=-3, limit=-2)

    # Then: Should return all trackers (both negative values ignored)
    assert len(trackers) == 5
    assert_all_trackers_have_user_id(trackers, user_id)
