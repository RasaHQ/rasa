import uuid
import warnings
from typing import List
from unittest.mock import Mock

import fakeredis
import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.tracker_stores.redis_tracker_store import (
    DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX,
    RedisTrackerStore,
)
from rasa.core.tracker_stores.tracker_store import (
    TrackerStore,
    check_if_tracker_store_async,
    create_tracker_store,
)
from rasa.plugin import plugin_manager
from rasa.shared.core.constants import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_SESSION_START_NAME,
)
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
from rasa.utils.endpoints import read_endpoint_config
from tests.utilities import filter_logs


def test_create_tracker_store_from_endpoint_config(
    domain: Domain, endpoints_path: str, monkeypatch: MonkeyPatch
):
    store = read_endpoint_config(endpoints_path, "tracker_store")
    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        username="username",
        password="password",
        record_exp=3000,
        use_ssl=True,
        ssl_keyfile="keyfile.key",
        ssl_certfile="certfile.crt",
        ssl_ca_certs="my-bundle.ca-bundle",
    )

    def mock_create_tracker_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        plugin_manager().hook, "create_tracker_store", mock_create_tracker_store
    )

    assert isinstance(tracker_store, type(TrackerStore.create(store, domain)))


def test_redis_tracker_store_invalid_key_prefix(domain: Domain):
    test_invalid_key_prefix = "$$ &!"

    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        key_prefix=test_invalid_key_prefix,
        record_exp=3000,
    )

    assert tracker_store._get_key_prefix() == DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX


def test_redis_tracker_store_valid_key_prefix(domain: Domain):
    test_valid_key_prefix = "spanish"

    tracker_store = RedisTrackerStore(
        domain=domain,
        host="localhost",
        port=6379,
        db=0,
        password="password",
        key_prefix=test_valid_key_prefix,
        record_exp=3000,
    )

    assert (
        tracker_store._get_key_prefix()
        == f"{test_valid_key_prefix}:{DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX}"
    )


def test_exception_tracker_store_from_endpoint_config(
    domain: Domain, monkeypatch: MonkeyPatch, endpoints_path: str
):
    """Check if tracker store properly handles exceptions.

    If we can not create a tracker store by instantiating the
    expected type (e.g. due to an exception) we should fallback to
    the default `InMemoryTrackerStore`."""

    store = read_endpoint_config(endpoints_path, "tracker_store")
    mock = Mock(side_effect=Exception("test exception"))
    monkeypatch.setattr(
        "rasa.core.tracker_stores.redis_tracker_store.RedisTrackerStore", mock
    )

    with pytest.raises(Exception) as e:
        TrackerStore.create(store, domain)

    assert "test exception" in str(e.value)


def test_raise_connection_exception_redis_tracker_store_creation(
    domain: Domain, monkeypatch: MonkeyPatch, endpoints_path: str
):
    store = read_endpoint_config(endpoints_path, "tracker_store")
    monkeypatch.setattr(
        "rasa.core.tracker_stores.redis_tracker_store.RedisTrackerStore",
        Mock(side_effect=ConnectionError()),
    )

    with pytest.raises(ConnectionException):
        TrackerStore.create(store, domain)


class MockedRedisTrackerStore(RedisTrackerStore):
    def __init__(
        self,
        domain: Domain,
    ) -> None:
        self.red = fakeredis.FakeStrictRedis()
        self.key_prefix = DEFAULT_REDIS_TRACKER_STORE_KEY_PREFIX
        self.record_exp = None
        super(RedisTrackerStore, self).__init__(domain, None)


async def test_redis_tracker_store_retrieve_full_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve_full_tracker(sender_id)
    assert tracker == tracker_with_restarted_event


async def test_redis_tracker_store_retrieve(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id

    await tracker_store.save(tracker_with_restarted_event)

    tracker = await tracker_store.retrieve(sender_id)
    assert list(tracker.events) == events_after_restart


async def test_redis_tracker_store_merge_trackers_same_session() -> None:
    start_session_sequence = [
        ActionExecuted(ACTION_SESSION_START_NAME),
        SessionStarted(),
        ActionExecuted(ACTION_LISTEN_NAME),
    ]
    events: List[Event] = start_session_sequence + [UserUttered("hello")]
    prior_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    events += [BotUttered("Hey! How can I help you?")]

    new_tracker = DialogueStateTracker.from_events(
        "same-session",
        evts=events,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    assert actual_tracker == new_tracker


def test_redis_tracker_store_merge_trackers_overlapping_session() -> None:
    prior_tracker_events: List[Event] = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
        SessionStarted(timestamp=2),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
        UserUttered("hello", timestamp=4),
        BotUttered("Hey! How can I help you?", timestamp=5),
        UserUttered("/restart", timestamp=6),
        ActionExecuted(ACTION_RESTART_NAME, timestamp=7),
    ]

    new_start_session = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=8),
        SessionStarted(timestamp=9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=10),
    ]

    prior_tracker_events += new_start_session
    prior_tracker = DialogueStateTracker.from_events(
        "overlapping-session",
        evts=prior_tracker_events,
    )

    after_restart_event = [UserUttered("hi again", timestamp=11)]
    new_tracker_events = new_start_session + after_restart_event

    new_tracker = DialogueStateTracker.from_events(
        "overlapping-session",
        evts=new_tracker_events,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    expected_events = prior_tracker_events + after_restart_event

    assert list(actual_tracker.events) == expected_events


def test_redis_tracker_store_merge_trackers_different_session() -> None:
    prior_tracker_events: List[Event] = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=1),
        SessionStarted(timestamp=2),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=3),
        UserUttered("hello", timestamp=4),
        BotUttered("Hey! How can I help you?", timestamp=5),
    ]
    prior_tracker = DialogueStateTracker.from_events(
        "different-session",
        evts=prior_tracker_events,
    )

    new_session = [
        ActionExecuted(ACTION_SESSION_START_NAME, timestamp=8),
        SessionStarted(timestamp=9),
        ActionExecuted(ACTION_LISTEN_NAME, timestamp=10),
        UserUttered("I need help.", timestamp=11),
    ]

    new_tracker = DialogueStateTracker.from_events(
        "different-session",
        evts=new_session,
    )

    actual_tracker = RedisTrackerStore._merge_trackers(prior_tracker, new_tracker)

    expected_events = prior_tracker_events + new_session
    assert list(actual_tracker.events) == expected_events


class HostExampleTrackerStore(RedisTrackerStore):
    pass


def test_tracker_store_with_host_argument_from_string(
    domain: Domain, monkeypatch: MonkeyPatch
):
    endpoints_path = "data/test_endpoints/custom_tracker_endpoints.yml"
    store_config = read_endpoint_config(endpoints_path, "tracker_store")
    store_config.type = (
        "tests.core.tracker_stores.test_redis_tracker_store.HostExampleTrackerStore"
    )

    def mock_create_tracker_store(*args, **kwargs):
        return None

    monkeypatch.setattr(
        plugin_manager().hook, "create_tracker_store", mock_create_tracker_store
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("error")
        tracker_store = TrackerStore.create(store_config, domain)

    assert len(record) == 0

    assert isinstance(tracker_store, HostExampleTrackerStore)


def test_create_tracker_store_from_endpoints_file_redis_tracker_store(
    domain: Domain,
) -> None:
    endpoint_config = read_endpoint_config(
        "data/test_endpoints/endpoints_redis.yml", "tracker_store"
    )
    tracker_store = create_tracker_store(endpoint_config, domain)

    assert check_if_tracker_store_async(tracker_store) is True
    assert isinstance(tracker_store, RedisTrackerStore)


async def test_redis_tracker_store_delete_tracker(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
    events_after_restart: List[Event],
) -> None:
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    # When
    with capture_logs() as caplog:
        await tracker_store.delete(sender_id)
        logs = filter_logs(
            caplog,
            event="redis_tracker_store.delete.deleted_tracker",
            log_level="info",
        )

        assert len(logs) == 1
    # Then
    tracker = await tracker_store.retrieve(sender_id)
    assert tracker is None


async def test_redis_tracker_store_delete_no_tracker(
    domain: Domain,
) -> None:
    with capture_logs() as caplog:
        tracker_store = MockedRedisTrackerStore(domain)
        sender_id = uuid.uuid4().hex
        await tracker_store.delete(sender_id)
        logs = filter_logs(
            caplog,
            event="redis_tracker_store.delete.no_tracker_for_sender_id",
            log_level="info",
            log_message_parts=[
                f"Could not find tracker for conversation ID '{sender_id}'."
            ],
        )

        assert len(logs) == 1
