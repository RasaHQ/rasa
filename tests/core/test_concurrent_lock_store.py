from pathlib import Path
from typing import List, Optional, Union
from unittest.mock import Mock, patch

import pytest
import structlog.testing
from _pytest.logging import LogCaptureFixture
from redis.exceptions import DataError

import rasa.utils.endpoints
from rasa.core.concurrent_lock_store import (
    DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX,
    ConcurrentRedisLockStore,
)
from rasa.core.lock_store import LockStore
from rasa.core.redis_connection_factory import DeploymentMode
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.exceptions import RasaException
from tests.utilities import filter_logs


def test_create_concurrent_redis_lock_store(tmp_path: Path) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    lock_store = LockStore.create(endpoint_config)

    assert isinstance(lock_store, ConcurrentRedisLockStore)


def test_create_concurrent_redis_lock_store_valid_custom_key_prefix(
    tmp_path: Path,
) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    custom_prefix = "testPrefix"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
            key_prefix: {custom_prefix}
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    with structlog.testing.capture_logs() as caplog:
        lock_store = LockStore.create(endpoint_config)

        logs = filter_logs(
            caplog,
            "concurrent_redis_lock_store._set_key_prefix.non_default_key_prefix",
            "debug",
        )
        assert len(logs) == 1

    assert isinstance(lock_store, ConcurrentRedisLockStore)
    assert (
        lock_store.key_prefix
        == custom_prefix + ":" + DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX
    )


def test_create_concurrent_redis_lock_store_invalid_custom_key_preifx(
    tmp_path: Path, caplog: LogCaptureFixture
) -> None:
    endpoints_file = tmp_path / "endpoints.yml"
    invalid_prefix = "test_prefix"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
            key_prefix: {invalid_prefix}
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    with structlog.testing.capture_logs() as caplog:
        lock_store = LockStore.create(endpoint_config)

        logs = filter_logs(
            caplog,
            "concurrent_redis_lock_store._set_key_prefix.default_instead_of_invalid_key_prefix",
            "warning",
        )
        assert len(logs) == 1

    assert isinstance(lock_store, ConcurrentRedisLockStore)
    assert lock_store.key_prefix == DEFAULT_CONCURRENT_REDIS_LOCK_STORE_KEY_PREFIX


@pytest.mark.parametrize(
    "deployment_mode,endpoints_yaml,sentinel_yaml,expected_mode,expected_endpoints,expected_sentinel_service",
    [
        ("standard", "", "", DeploymentMode.STANDARD.value, None, None),
        (
            "cluster",
            """
                - node1:6379
                - node2:6379
            """,
            "",
            DeploymentMode.CLUSTER.value,
            ["node1:6379", "node2:6379"],
            None,
        ),
        (
            "sentinel",
            """
                - sentinel1:26379
                - sentinel2:26379
                - sentinel3:26379
            """,
            "sentinel_service: custom",
            DeploymentMode.SENTINEL.value,
            ["sentinel1:26379", "sentinel2:26379", "sentinel3:26379"],
            "custom",
        ),
    ],
)
def test_create_concurrent_redis_lock_store_high_availability_modes(
    deployment_mode: str,
    endpoints_yaml: str,
    sentinel_yaml: str,
    expected_mode: str,
    expected_endpoints: Optional[List[str]],
    expected_sentinel_service: Optional[str],
    tmp_path: Path,
):
    """Test concurrent lock store creation with different high availability modes."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        endpoints_file = tmp_path / "endpoints.yml"
        endpoints_file.write_text(
            f"""
            version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
            lock_store:
                type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
                host: localhost
                port: 6379
                username: username
                password: password
                deployment_mode: {deployment_mode}
                endpoints:
                    {endpoints_yaml}
                {sentinel_yaml}
            """
        )

        endpoint_config = rasa.utils.endpoints.read_endpoint_config(
            str(endpoints_file), "lock_store"
        )

        # When
        lock_store = ConcurrentRedisLockStore(endpoint_config=endpoint_config)

        # Then
        assert isinstance(lock_store, ConcurrentRedisLockStore)
        assert lock_store.red == mock_redis
        mock_create.assert_called_once()

        call_args = mock_create.call_args[0][0]

        assert call_args.deployment_mode == expected_mode
        assert call_args.endpoints == expected_endpoints
        assert call_args.sentinel_service == expected_sentinel_service


def test_create_concurrent_redis_lock_store_default_deployment_mode(tmp_path: Path):
    """Test concurrent lock store creation with default deployment mode."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        endpoints_file = tmp_path / "endpoints.yml"
        endpoints_file.write_text(
            f"""
            version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
            lock_store:
                type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
                host: localhost
                port: 6379
            """
        )

        endpoint_config = rasa.utils.endpoints.read_endpoint_config(
            str(endpoints_file), "lock_store"
        )
        # When
        lock_store = ConcurrentRedisLockStore(endpoint_config=endpoint_config)

        # Then
        assert isinstance(lock_store, ConcurrentRedisLockStore)
        assert lock_store.red == mock_redis
        mock_create.assert_called_once()

        call_args = mock_create.call_args[0][0]
        print(call_args)

        assert call_args.deployment_mode == DeploymentMode.STANDARD.value


def test_create_concurrent_redis_lock_store_invalid_configuration(tmp_path: Path):
    """Test concurrent lock store creation with invalid configuration."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        f"""
        version: {LATEST_TRAINING_DATA_FORMAT_VERSION}
        lock_store:
            type: rasa.core.concurrent_lock_store.ConcurrentRedisLockStore
            host: localhost
            port: 6379
            username: username
            password: password
            endpoints:
                - 123
        """
    )
    endpoint_config = rasa.utils.endpoints.read_endpoint_config(
        str(endpoints_file), "lock_store"
    )
    with pytest.raises(RasaException) as exc_info:
        ConcurrentRedisLockStore(endpoint_config=endpoint_config)
    assert "Invalid Redis configuration" in str(exc_info.value)


@pytest.mark.parametrize(
    "redis_response",
    [
        # bytes response (needs conversion)
        b'{"number": 1, "expires": 1234567900.0}',
        # string response (no conversion needed)
        '{"number": 1, "expires": 1234567900.0}',
    ],
)
def test_get_lock_handles_bytes_and_string_responses(redis_response: Union[bytes, str]):
    """Test that get_lock properly handles both bytes and string responses."""
    conversation_id = "test_conversation"
    mock_redis = Mock()

    mock_redis.keys.return_value = [f"rasa:lock_store:{conversation_id}:1"]
    mock_redis.get.return_value = redis_response

    endpoint_config = Mock()
    endpoint_config.kwargs = {}

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection",
        return_value=mock_redis,
    ):
        lock_store = ConcurrentRedisLockStore(endpoint_config)

    result = lock_store.get_lock(conversation_id)

    assert result is not None
    assert result.conversation_id == conversation_id
    assert len(result.tickets) == 1

    # Verify the ticket was parsed correctly regardless of bytes/string
    ticket = result.tickets[0]
    assert ticket.number == 1
    assert ticket.expires == 1234567900.0


def test_get_keys_by_pattern_handles_scan_exception():
    """Test that _get_keys_by_pattern handles SCAN exceptions in cluster mode."""
    conversation_id = "test_conversation"

    # Mock the lock store with cluster deployment mode
    endpoint_config = Mock()
    endpoint_config.kwargs = {"deployment_mode": "cluster"}

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        mock_redis = Mock()
        mock_create.return_value = mock_redis
        lock_store = ConcurrentRedisLockStore(endpoint_config)

    # First SCAN call returns some keys, second call raises DataError
    mock_redis.scan.side_effect = [
        (1, ["key1", "key2"]),
        DataError("Invalid input of type: 'dict'"),
    ]

    pattern = lock_store.key_prefix + conversation_id + ":[0-9]*"

    with structlog.testing.capture_logs() as caplog:
        result = lock_store._get_keys_by_pattern(pattern)

    assert result == ["key1", "key2"]
    assert mock_redis.scan.call_count == 2

    debug_logs = filter_logs(
        caplog,
        "concurrent_redis_lock_store._get_keys_by_pattern.scan_interrupted",
        "warning",
    )
    assert len(debug_logs) == 1
    assert "SCAN interrupted in cluster mode" in debug_logs[0]["event_info"]
    assert "2 keys found so far" in debug_logs[0]["event_info"]


def test_get_keys_by_pattern_handles_dict_cursor_in_cluster_mode():
    """Test that _get_keys_by_pattern handles dict cursor properly in cluster mode."""
    conversation_id = "test_conversation"
    endpoint_config = Mock()
    endpoint_config.kwargs = {"deployment_mode": "cluster"}

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        mock_redis = Mock()
        mock_create.return_value = mock_redis
        lock_store = ConcurrentRedisLockStore(endpoint_config)

    # Mock Redis returning dict cursor
    mock_redis.scan.side_effect = [
        (
            {"127.0.0.1:7000": 1, "127.0.0.1:7001": 1, "127.0.0.1:7002": 1},
            ["key1", "key2"],
        ),  # First call with active cursors
        (
            {"127.0.0.1:7000": 0, "127.0.0.1:7001": 0, "127.0.0.1:7002": 0},
            ["key3"],
        ),  # Second call - all done
    ]

    with structlog.testing.capture_logs() as caplog:
        pattern = lock_store.key_prefix + conversation_id + ":[0-9]*"
        result = lock_store._get_keys_by_pattern(pattern)

    assert result == ["key1", "key2", "key3"]
    assert mock_redis.scan.call_count == 2

    # Verify no scan_interrupted warnings occurred
    scan_warnings = filter_logs(
        caplog,
        "concurrent_redis_lock_store._get_keys_by_pattern.scan_interrupted",
        "warning",
    )
    assert len(scan_warnings) == 0


def test_get_keys_by_pattern_standard_mode_uses_keys():
    """Test that _get_keys_by_pattern uses KEYS command in standard mode."""
    conversation_id = "test_conversation"
    endpoint_config = Mock()
    endpoint_config.kwargs = {}

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        mock_redis = Mock()
        mock_create.return_value = mock_redis
        lock_store = ConcurrentRedisLockStore(endpoint_config)

    # Mock Redis KEYS response
    expected_keys = ["key1", "key2", "key3"]
    mock_redis.keys.return_value = expected_keys

    pattern = lock_store.key_prefix + conversation_id + ":[0-9]*"

    result = lock_store._get_keys_by_pattern(pattern)
    assert result == expected_keys

    # Verify KEYS was called, not SCAN
    mock_redis.keys.assert_called_once_with(pattern)
    mock_redis.scan.assert_not_called()
