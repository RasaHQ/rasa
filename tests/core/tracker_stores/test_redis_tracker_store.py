import time
import uuid
import warnings
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import fakeredis
import pytest
from pytest import MonkeyPatch
from structlog.testing import capture_logs

from rasa.core.constants import (
    ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME,
    IAM_CLOUD_PROVIDER_ENV_VAR_NAME,
)
from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSElasticacheRedisIAMCredentialsProvider,
)
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
    assert_all_trackers_have_user_id,
    assert_tracker_has_user_id,
    create_multiple_trackers_with_user_id,
    create_tracker_with_user_id,
    create_trackers_with_same_timestamp,
    old_tracker_gets_timestamp_on_save,
    old_tracker_gets_timestamp_on_update,
    sort_key,
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


# Helper functions for Redis tracker store tests
def create_redis_tracker_store_with_ttl(
    domain: Domain, record_exp: Optional[float]
) -> MockedRedisTrackerStore:
    """Create a Redis tracker store with a specific TTL.

    Args:
        domain: The domain for the tracker store.
        record_exp: TTL in seconds, or None for no expiration.

    Returns:
        A MockedRedisTrackerStore instance with the specified TTL.
    """
    tracker_store = MockedRedisTrackerStore(domain)
    tracker_store.record_exp = record_exp
    return tracker_store


def get_sorted_set_score(
    tracker_store: MockedRedisTrackerStore, user_id: str, sender_id: str
) -> Optional[float]:
    """Get the expiration timestamp score for a member in the sorted set.

    Args:
        tracker_store: The Redis tracker store.
        user_id: The user ID.
        sender_id: The sender ID.

    Returns:
        The score (expiration timestamp) or None if not found.
    """
    user_trackers_key = tracker_store._get_user_trackers_key(user_id)
    return tracker_store.red.zscore(user_trackers_key, sender_id)


def expire_sorted_set_member(
    tracker_store: MockedRedisTrackerStore,
    user_id: str,
    sender_id: str,
    seconds_ago: float,
) -> None:
    """Manually expire a sorted set member by setting its score to a past timestamp.

    Args:
        tracker_store: The Redis tracker store.
        user_id: The user ID.
        sender_id: The sender ID to expire.
        seconds_ago: How many seconds in the past to set the expiration.
    """
    user_trackers_key = tracker_store._get_user_trackers_key(user_id)
    past_time = time.time() - seconds_ago
    tracker_store.red.zadd(user_trackers_key, {sender_id: past_time})


def get_sorted_set_members(
    tracker_store: MockedRedisTrackerStore, user_id: str
) -> List[str]:
    """Get all members from a sorted set for a user.

    Args:
        tracker_store: The Redis tracker store.
        user_id: The user ID.

    Returns:
        List of sender IDs in the sorted set.
    """
    user_trackers_key = tracker_store._get_user_trackers_key(user_id)
    members = tracker_store.red.zrange(user_trackers_key, 0, -1)
    return [m.decode("utf-8") if isinstance(m, bytes) else m for m in members]


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


async def test_redis_tracker_store_create_with_iam_enabled(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv(ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME, "true")
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.StrictRedis", mock_redis)

    tracker_store = RedisTrackerStore(domain)
    assert isinstance(tracker_store, RedisTrackerStore)
    mock_redis.assert_called_once()
    assert mock_redis.call_args[1].get("credential_provider") is not None
    assert isinstance(
        mock_redis.call_args[1].get("credential_provider"),
        AWSElasticacheRedisIAMCredentialsProvider,
    )


async def test_redis_tracker_store_create_with_iam_disabled_for_elasticache(
    domain: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    # Given
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    mock_redis = MagicMock()
    monkeypatch.setattr("redis.StrictRedis", mock_redis)

    tracker_store = RedisTrackerStore(domain)
    assert isinstance(tracker_store, RedisTrackerStore)
    mock_redis.assert_called_once()
    assert mock_redis.call_args[1].get("credential_provider") is None


async def test_redis_tracker_store_get_trackers_by_user_id(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore.get_trackers_by_user_id returns correct trackers."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create trackers with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)
    await create_tracker_with_user_id(
        tracker_store, "sender2", user_id, [SessionStarted(), UserUttered("hi")]
    )

    # Create tracker with different user_id
    await create_tracker_with_user_id(
        tracker_store, "sender3", "user_456", [SessionStarted(), UserUttered("hey")]
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 2
    assert {t.sender_id for t in trackers} == {"sender1", "sender2"}
    for tracker in trackers:
        assert tracker.user_id == user_id


async def test_redis_tracker_store_get_trackers_by_user_id_no_matches(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore returns empty list when no matches exist."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create tracker with different user_id
    await create_tracker_with_user_id(tracker_store, "sender1", "user_456")

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 0


async def test_redis_tracker_store_get_trackers_by_user_id_filters_no_user_id(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore filters out trackers without user_id."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

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
    assert trackers[0].sender_id == "sender1"
    assert trackers[0].user_id == user_id


async def test_redis_tracker_store_get_trackers_by_user_id_save_sets_user_id(
    domain: Domain,
) -> None:
    """Test that save method preserves user_id when saving trackers."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"
    sender_id = "sender1"

    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    await tracker_store.save(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], sender_id, user_id)


async def test_redis_tracker_store_get_trackers_by_user_id_update_sets_user_id(
    domain: Domain,
) -> None:
    """Test that update method preserves user_id when updating trackers."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"
    sender_id = "sender1"

    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    await tracker_store.save(tracker)

    tracker.update(UserUttered("Hello again!"))
    await tracker_store.update(tracker)

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then
    assert len(trackers) == 1
    assert_tracker_has_user_id(trackers[0], sender_id, user_id)


async def test_redis_tracker_store_get_trackers_by_user_id_with_limit(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore.get_trackers_by_user_id respects limit parameter."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create multiple trackers with user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=5)

    # Then
    assert len(trackers) == 5
    for tracker in trackers:
        assert tracker.user_id == user_id

    assert trackers == sorted(saved_trackers, key=sort_key)[:5]


async def test_redis_tracker_store_get_trackers_by_user_id_with_skip(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore.get_trackers_by_user_id respects skip parameter."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create multiple trackers with user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=3)

    # Then
    assert len(trackers) == 7  # 10 total - 3 skipped
    for tracker in trackers:
        assert tracker.user_id == user_id

    assert trackers == sorted(saved_trackers, key=sort_key)[3:]


async def test_redis_tracker_store_get_trackers_by_user_id_with_skip_and_limit(
    domain: Domain,
) -> None:
    """Test RedisTrackerStore.get_trackers_by_user_id respects both skip and limit."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create multiple trackers with user_id
    saved_trackers = await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 10
    )

    # When
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=2, limit=3)

    # Then
    assert len(trackers) == 3
    assert_all_trackers_have_user_id(trackers, user_id)

    saved_trackers_sorted = sorted(saved_trackers, key=sort_key)
    assert trackers == saved_trackers_sorted[2:5]


async def test_redis_tracker_store_get_trackers_by_user_id_deserialization_error(
    domain: Domain,
) -> None:
    """Test get_trackers_by_user_id handles deserialization errors gracefully."""
    # Given
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create 2 valid trackers with user_id
    await create_tracker_with_user_id(tracker_store, "sender1", user_id)
    await create_tracker_with_user_id(tracker_store, "sender2", user_id)

    # Modify the stored data for one tracker to be invalid JSON
    tracker_store.red.set(tracker_store.key_prefix + "sender2", "invalid json")

    # When
    with capture_logs() as caplog:
        trackers = await tracker_store.get_trackers_by_user_id(user_id)

        # Then
        assert len(trackers) == 1
        assert_tracker_has_user_id(trackers[0], "sender1", user_id)

        # Verify error was logged
        logs = filter_logs(
            caplog,
            event="redis_tracker_store.get_trackers_by_user_id.deserialization_failed",
            log_level="error",
            log_message_parts=["Failed to deserialize tracker for sender_id 'sender2'"],
        )
        assert len(logs) == 1


# Backward compatibility tests for conversation_started_timestamp
@pytest.mark.asyncio
async def test_redis_old_tracker_gets_timestamp_on_save(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it on save."""
    tracker_store = MockedRedisTrackerStore(domain)
    await old_tracker_gets_timestamp_on_save(tracker_store, domain=domain)


@pytest.mark.asyncio
async def test_redis_old_tracker_gets_timestamp_on_update(domain: Domain) -> None:
    """Test that old tracker without conversation_started_timestamp gets it
    on update."""
    tracker_store = MockedRedisTrackerStore(domain)
    await old_tracker_gets_timestamp_on_update(tracker_store, domain=domain)


# Sorting consistency tests
@pytest.mark.asyncio
async def test_redis_sorting_by_sender_id_when_timestamps_identical(
    domain: Domain,
) -> None:
    """Test that trackers with identical timestamps are sorted by sender_id."""
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"
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
async def test_redis_pagination_very_large_skip(domain: Domain) -> None:
    """Test that very large skip values are handled gracefully."""
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large skip should return empty list
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=1000000)

    assert len(trackers) == 0


@pytest.mark.asyncio
async def test_redis_pagination_very_large_limit(domain: Domain) -> None:
    """Test that very large limit values are handled gracefully."""
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # Very large limit should return all items (up to available)
    trackers = await tracker_store.get_trackers_by_user_id(user_id, limit=1000000)

    assert len(trackers) == 5


@pytest.mark.asyncio
async def test_redis_sorted_set_member_gets_expiration_timestamp_with_ttl(
    domain: Domain,
) -> None:
    """Test that sorted set members get expiration timestamps when TTL is set."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 3600)  # 1 hour TTL
    user_id = "user_123"
    sender_id = "sender_1"

    # When: Save a tracker with TTL
    await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Then: Check that the sorted set member has an expiration timestamp
    current_time = time.time()
    score = get_sorted_set_score(tracker_store, user_id, sender_id)

    assert score is not None
    # Score should be approximately current_time + TTL (within 1 second tolerance)
    expected_expiration = current_time + 3600.0
    assert abs(score - expected_expiration) < 1.0
    # Score should be greater than current time (not expired)
    assert score > current_time


@pytest.mark.asyncio
async def test_redis_sorted_set_member_uses_inf_without_ttl(domain: Domain) -> None:
    """Test that sorted set members use +inf score when no TTL is set."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, None)  # No TTL
    user_id = "user_123"
    sender_id = "sender_1"

    # When: Save a tracker without TTL
    await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Then: Check that the sorted set member has +inf score
    score = get_sorted_set_score(tracker_store, user_id, sender_id)

    assert score is not None
    assert score == float("inf")


@pytest.mark.asyncio
async def test_redis_sorted_set_member_uses_custom_timeout(domain: Domain) -> None:
    """Test that sorted set members use custom timeout when provided to save()."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 3600)  # Default 1 hour
    custom_timeout = 1800  # Custom 30 minutes
    user_id = "user_123"
    sender_id = "sender_1"

    # When: Save a tracker with custom timeout
    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    await tracker_store.save(tracker, timeout=custom_timeout)

    # Then: Check that the sorted set member uses custom timeout
    current_time = time.time()
    score = get_sorted_set_score(tracker_store, user_id, sender_id)

    assert score is not None
    # Score should be approximately current_time + custom_timeout
    expected_expiration = current_time + custom_timeout
    assert abs(score - expected_expiration) < 1.0


@pytest.mark.asyncio
async def test_redis_expired_members_are_cleaned_up_on_retrieval(
    domain: Domain,
) -> None:
    """Test that expired sorted set members are cleaned up when retrieving trackers."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 1)  # 1 second TTL
    user_id = "user_123"
    sender_id_expired = "sender_expired"
    sender_id_valid = "sender_valid"

    # Create a tracker that will expire
    await create_tracker_with_user_id(tracker_store, sender_id_expired, user_id)

    # Manually expire the member (simulating expired member)
    expire_sorted_set_member(tracker_store, user_id, sender_id_expired, 10)

    # Create a valid tracker
    await create_tracker_with_user_id(tracker_store, sender_id_valid, user_id)

    # When: Retrieve trackers
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then: Expired member should be removed from sorted set
    members = get_sorted_set_members(tracker_store, user_id)
    assert sender_id_expired not in members
    # Only valid tracker should be returned
    assert len(trackers) == 1
    assert trackers[0].sender_id == sender_id_valid


@pytest.mark.asyncio
async def test_redis_expired_members_not_returned(domain: Domain) -> None:
    """Test that expired sorted set members are not returned in results."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 1)  # 1 second TTL
    user_id = "user_123"
    sender_id_expired = "sender_expired"
    sender_id_valid = "sender_valid"

    # Create trackers
    await create_tracker_with_user_id(tracker_store, sender_id_expired, user_id)
    await create_tracker_with_user_id(tracker_store, sender_id_valid, user_id)

    # Manually expire one member
    expire_sorted_set_member(tracker_store, user_id, sender_id_expired, 5)

    # When: Retrieve trackers
    trackers = await tracker_store.get_trackers_by_user_id(user_id)

    # Then: Only non-expired tracker should be returned
    assert len(trackers) == 1
    assert trackers[0].sender_id == sender_id_valid
    assert trackers[0].user_id == user_id


@pytest.mark.asyncio
async def test_redis_multiple_trackers_different_ttls(domain: Domain) -> None:
    """Test that multiple trackers with different TTLs are handled correctly."""
    tracker_store = MockedRedisTrackerStore(domain)  # No default TTL
    user_id = "user_123"
    sender_id_1 = "sender_1"
    sender_id_2 = "sender_2"
    sender_id_3 = "sender_3"

    # Create trackers with different timeouts
    tracker1 = await create_tracker_with_user_id(tracker_store, sender_id_1, user_id)
    await tracker_store.save(tracker1, timeout=3600)  # 1 hour

    tracker2 = await create_tracker_with_user_id(tracker_store, sender_id_2, user_id)
    await tracker_store.save(tracker2, timeout=1800)  # 30 minutes

    await create_tracker_with_user_id(tracker_store, sender_id_3, user_id)
    # No timeout (uses record_exp which is None, so +inf)

    # When: Check sorted set scores
    current_time = time.time()
    score1 = get_sorted_set_score(tracker_store, user_id, sender_id_1)
    score2 = get_sorted_set_score(tracker_store, user_id, sender_id_2)
    score3 = get_sorted_set_score(tracker_store, user_id, sender_id_3)

    # Then: All should have correct expiration timestamps
    assert score1 is not None
    assert score2 is not None
    assert score3 is not None

    # Score1 should expire after score2 (longer TTL)
    assert score1 > score2
    # Score3 should be +inf (no TTL)
    assert score3 == float("inf")
    # All should be valid (not expired)
    assert score1 > current_time
    assert score2 > current_time

    # All trackers should be retrievable
    trackers = await tracker_store.get_trackers_by_user_id(user_id)
    assert len(trackers) == 3


@pytest.mark.asyncio
async def test_redis_update_refreshes_expiration_timestamp(domain: Domain) -> None:
    """Test that updating a tracker refreshes its expiration timestamp."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 3600)  # 1 hour TTL
    user_id = "user_123"
    sender_id = "sender_1"

    # Create a tracker
    tracker = await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Get initial expiration
    initial_score = get_sorted_set_score(tracker_store, user_id, sender_id)
    assert initial_score is not None

    # Wait a bit and update the tracker
    time.sleep(0.1)
    await tracker_store.update(tracker)

    # Then: Expiration should be refreshed (new score should be later)
    updated_score = get_sorted_set_score(tracker_store, user_id, sender_id)
    assert updated_score is not None
    assert updated_score > initial_score


@pytest.mark.asyncio
async def test_redis_delete_removes_from_sorted_set(domain: Domain) -> None:
    """Test that deleting a tracker removes it from the sorted set."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 3600)
    user_id = "user_123"
    sender_id = "sender_1"

    # Create a tracker
    await create_tracker_with_user_id(tracker_store, sender_id, user_id)

    # Verify it's in the sorted set
    assert get_sorted_set_score(tracker_store, user_id, sender_id) is not None

    # When: Delete the tracker
    await tracker_store.delete(sender_id)

    # Then: It should be removed from the sorted set
    assert get_sorted_set_score(tracker_store, user_id, sender_id) is None


@pytest.mark.asyncio
async def test_redis_cleanup_logs_when_expired_members_removed(
    domain: Domain,
) -> None:
    """Test that cleanup of expired members is logged."""
    tracker_store = create_redis_tracker_store_with_ttl(domain, 1)
    user_id = "user_123"
    sender_id = "sender_expired"

    await create_tracker_with_user_id(tracker_store, sender_id, user_id)
    expire_sorted_set_member(tracker_store, user_id, sender_id, 10)

    with capture_logs() as caplog:
        await tracker_store.get_trackers_by_user_id(user_id)

        logs = filter_logs(
            caplog,
            event=(
                "redis_tracker_store.get_trackers_by_user_id." "cleaned_expired_members"
            ),
            log_level="debug",
            log_message_parts=["Cleaned up 1 expired sender_ids"],
        )
        assert len(logs) == 1


@pytest.mark.asyncio
async def test_redis_negative_skip_and_limit_ignored(domain: Domain) -> None:
    """Test that both negative skip and limit values are ignored."""
    tracker_store = MockedRedisTrackerStore(domain)
    user_id = "user_123"

    # Create some trackers
    await create_multiple_trackers_with_user_id(
        tracker_store, user_id, 5, domain=domain
    )

    # When: Retrieve with both negative skip and limit
    trackers = await tracker_store.get_trackers_by_user_id(user_id, skip=-3, limit=-2)

    # Then: Should return all trackers (both negative values ignored)
    assert len(trackers) == 5
    assert_all_trackers_have_user_id(trackers, user_id)
