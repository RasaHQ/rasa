import uuid
import warnings
from typing import List
from unittest.mock import Mock, patch

import fakeredis
import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.redis_connection_factory import DeploymentMode, RedisConfig
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
from rasa.shared.exceptions import ConnectionException, RasaException
from rasa.utils.endpoints import read_endpoint_config
from tests.core.tracker_stores.conftest import (
    _saved_tracker_with_multiple_session_starts,
)
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


@pytest.mark.parametrize(
    "deployment_mode,endpoints,sentinel_service,expected_mode,expected_endpoints,expected_sentinel_service",
    [
        ("standard", None, None, DeploymentMode.STANDARD.value, None, None),
        (
            "cluster",
            ["node1:6379", "node2:6379"],
            None,
            DeploymentMode.CLUSTER.value,
            ["node1:6379", "node2:6379"],
            None,
        ),
        (
            "sentinel",
            ["sentinel1:26379"],
            "custom",
            DeploymentMode.SENTINEL.value,
            ["sentinel1:26379"],
            "custom",
        ),
    ],
)
def test_create_tracker_store_high_availability_modes(
    domain: Domain,
    deployment_mode,
    endpoints,
    sentinel_service,
    expected_mode,
    expected_endpoints,
    expected_sentinel_service,
):
    """Test tracker store creation with different high availability modes."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        # When
        tracker_store = RedisTrackerStore(
            domain=domain,
            host="localhost",
            port=6379,
            db=0,
            deployment_mode=deployment_mode,
            endpoints=endpoints,
            sentinel_service=sentinel_service,
        )

        # Then
        assert isinstance(tracker_store, RedisTrackerStore)
        assert tracker_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]
        assert isinstance(config, RedisConfig)

        assert config.deployment_mode == expected_mode
        assert config.endpoints == expected_endpoints
        assert config.sentinel_service == expected_sentinel_service


def test_create_tracker_store_default_deployment_mode(domain: Domain):
    """Test tracker store creation with standard deployment mode."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        # When
        tracker_store = RedisTrackerStore(
            domain=domain,
            host="localhost",
            port=6379,
            db=0,
        )

        # Then
        assert isinstance(tracker_store, RedisTrackerStore)
        assert tracker_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]

        assert isinstance(config, RedisConfig)
        assert config.deployment_mode == DeploymentMode.STANDARD.value
        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"host": 123, "port": 6379},
        {"endpoints": [123, "localhost:6379"]},
    ],
)
def test_redis_tracker_store_validation_error(domain: Domain, invalid_config):
    """Test that RedisTrackerStore properly handles configuration validation errors."""

    with pytest.raises(RasaException) as exc_info:
        RedisTrackerStore(domain=domain, **invalid_config)

    assert "Invalid Redis configuration" in str(exc_info.value)


def test_redis_tracker_store_invalid_key_prefix(domain: Domain):
    test_invalid_key_prefix = "$$ &!"

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
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

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
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
    the default `InMemoryTrackerStore`.
    """
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


async def test_redis_tracker_store_delete_tracker_with_prefix(
    domain: Domain,
    tracker_with_restarted_event: DialogueStateTracker,
) -> None:
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    sender_id = tracker_with_restarted_event.sender_id
    await tracker_store.save(tracker_with_restarted_event)

    ## This will retrieve the sender_id with the prefix as bytes object
    existing_sender_ids = await tracker_store.keys()
    assert len(list(existing_sender_ids)) == 1
    existing_sender_id = next(iter(existing_sender_ids))
    existing_sender_id = existing_sender_id.decode("utf-8")
    assert existing_sender_id == f"{tracker_store.key_prefix}{sender_id}"

    # When
    with capture_logs() as caplog:
        await tracker_store.delete(existing_sender_id)
        logs = filter_logs(
            caplog,
            event="redis_tracker_store.delete.deleted_tracker",
            log_level="info",
        )

        assert len(logs) == 1

    # Then
    tracker = await tracker_store.retrieve(existing_sender_id)
    assert tracker is None


async def test_redis_tracker_store_save_tracker_with_prefix(
    domain: Domain,
    events_after_restart: List[Event],
) -> None:
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    expected_sender_id = tracker_store.key_prefix + uuid.uuid4().hex
    tracker = DialogueStateTracker.from_events(expected_sender_id, events_after_restart)

    # When
    await tracker_store.save(tracker)

    # Then
    actual_sender_ids = await tracker_store.keys()
    assert len(list(actual_sender_ids)) == 1
    actual_sender_id = next(iter(actual_sender_ids))
    actual_sender_id = actual_sender_id.decode("utf-8")
    assert actual_sender_id == expected_sender_id


async def test_redis_tracker_store_update_tracker(domain: Domain) -> None:
    # Given
    sender_id = uuid.uuid4().hex
    tracker_store = MockedRedisTrackerStore(domain)
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


async def test_redis_tracker_store_update_tracker_with_prefix(domain: Domain) -> None:
    # Given
    sender_id = uuid.uuid4().hex
    tracker_store = MockedRedisTrackerStore(domain)
    tracker = await _saved_tracker_with_multiple_session_starts(
        tracker_store, sender_id
    )
    new_events = list(tracker.events)[3:] + [
        UserUttered("What's the weather like today?")
    ]
    new_tracker = DialogueStateTracker.from_events(
        tracker_store.key_prefix + sender_id,
        new_events,
    )

    # When
    await tracker_store.update(new_tracker)

    # Then
    updated_tracker = await tracker_store.retrieve(sender_id)
    assert updated_tracker.events == new_tracker.events
