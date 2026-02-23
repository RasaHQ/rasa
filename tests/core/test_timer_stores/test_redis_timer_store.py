import time
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from structlog.testing import capture_logs

from rasa.core.redis_connection_factory import DeploymentMode
from rasa.core.timer_stores.redis_timer_store import (
    RedisSessionTimerStore,
    RedisSessionTimerStoreConfig,
)
from rasa.core.timer_stores.timer_store import SessionTimerStore
from rasa.utils.endpoints import read_endpoint_config
from tests.core.timer_test_helpers import assert_timer, timer_json_bytes
from tests.utilities import filter_logs


@pytest.fixture
def endpoints_path(tmp_path):
    """Create a temporary endpoints file for testing."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        """
timer_store:
  type: redis
  url: localhost
  port: 6379
  db: 2
  username: username
  password: password
  key_prefix: timer
  poll_interval: 2.5
  use_ssl: True
  ssl_keyfile: "keyfile.key"
  ssl_certfile: "certfile.crt"
  ssl_ca_certs: "my-bundle.ca-bundle"
"""
    )
    return str(endpoints_file)


@pytest.fixture
def redis_timer_store(mock_redis) -> RedisSessionTimerStore:
    """Create a Redis timer store with mocked connection."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection",
        return_value=mock_redis,
    ):
        config = RedisSessionTimerStoreConfig(host="localhost", port=6379, db=2)
        return RedisSessionTimerStore(config=config)


def test_redis_timer_store_config_validates():
    """Test that RedisSessionTimerStoreConfig validates correctly."""
    config = RedisSessionTimerStoreConfig(
        host="localhost",
        port=6379,
        db=2,
        key_prefix="myapp",
    )

    assert config.type == "redis"
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 2
    assert config.key_prefix == "myapp"
    assert config.socket_timeout == 10  # Default value
    assert config.poll_interval == 1.0  # Default value


def test_redis_timer_store_config_url_maps_to_host():
    """Test that deprecated 'url' field is mapped to 'host'."""
    with pytest.warns(FutureWarning, match="url.*deprecated.*host"):
        config = RedisSessionTimerStoreConfig(
            url="my-redis-host",
            port=6379,
            db=2,
        )

    assert config.host == "my-redis-host"


def test_redis_timer_store_config_url_and_host_raises():
    """Test that providing both 'url' and 'host' raises an exception."""
    from rasa.shared.exceptions import RasaException

    with pytest.raises(RasaException, match="cannot specify both 'url' and 'host'"):
        RedisSessionTimerStoreConfig(
            url="my-redis-host",
            host="localhost",
            port=6379,
            db=2,
        )


def test_redis_timer_store_config_poll_interval_customizable():
    """Test that poll_interval can be customized via config."""
    config = RedisSessionTimerStoreConfig(
        host="localhost",
        port=6379,
        poll_interval=5.0,
    )

    assert config.poll_interval == 5.0


def test_redis_timer_store_config_from_endpoint_dict(endpoints_path: str):
    """Test that poll_interval is read from endpoint config."""
    endpoint_config = read_endpoint_config(endpoints_path, endpoint_type="timer_store")
    config = RedisSessionTimerStoreConfig.model_validate(endpoint_config.to_dict())

    assert config.poll_interval == 2.5  # Value from endpoints fixture


@pytest.mark.parametrize(
    "key_prefix,expected_prefix",
    [
        (None, "timer:"),  # Default
        ("myapp", "myapp:timer:"),  # Custom alphanumeric
        ("my_app:", "timer:"),  # Invalid (non-alphanumeric), falls back to default
    ],
)
def test_timer_store_key_prefix(key_prefix, expected_prefix):
    """Test that timer store handles key prefix correctly."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        config = RedisSessionTimerStoreConfig(
            host="localhost", port=6379, db=2, key_prefix=key_prefix
        )
        timer_store = RedisSessionTimerStore(config=config)

        assert timer_store.key_prefix == expected_prefix


@pytest.mark.parametrize(
    "key_prefix,expected_prefix",
    [
        (None, "{timer}:"),  # Default cluster prefix with hash tag
        ("myapp", "myapp:{timer}:"),  # Custom prefix with cluster hash tag
    ],
)
def test_timer_store_key_prefix_cluster_mode(key_prefix, expected_prefix):
    """Test that cluster mode uses hash tag in key prefix for slot co-location."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        config = RedisSessionTimerStoreConfig(
            host="localhost",
            port=6379,
            db=2,
            key_prefix=key_prefix,
            deployment_mode="cluster",
            endpoints=["node1:6379", "node2:6379"],
        )
        timer_store = RedisSessionTimerStore(config=config)

        assert timer_store.key_prefix == expected_prefix


@pytest.mark.parametrize(
    "extra_config,expected_mode,expected_endpoints,expected_sentinel_service",
    [
        ({"deployment_mode": "standard"}, DeploymentMode.STANDARD.value, None, None),
        (
            {"deployment_mode": "cluster", "endpoints": ["node1:6379", "node2:6379"]},
            DeploymentMode.CLUSTER.value,
            ["node1:6379", "node2:6379"],
            None,
        ),
        (
            {
                "deployment_mode": "sentinel",
                "endpoints": ["sentinel1:26379"],
                "sentinel_service": "mymaster",
            },
            DeploymentMode.SENTINEL.value,
            ["sentinel1:26379"],
            "mymaster",
        ),
        ({}, DeploymentMode.STANDARD.value, None, None),  # Default case
    ],
)
def test_create_timer_store_deployment_modes(
    extra_config: Dict[str, Any],
    expected_mode: str,
    expected_endpoints: Optional[List[str]],
    expected_sentinel_service: Optional[str],
):
    """Test timer store creation with different deployment modes including default."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ) as mock_create:
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        base_config = {"host": "localhost", "port": 6379, "db": 2}
        config = RedisSessionTimerStoreConfig(**{**base_config, **extra_config})

        timer_store = RedisSessionTimerStore(config=config)

        assert isinstance(timer_store, RedisSessionTimerStore)
        assert timer_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]

        assert config.deployment_mode == expected_mode
        assert config.endpoints == expected_endpoints
        assert config.sentinel_service == expected_sentinel_service


def test_create_timer_store_from_endpoint_config(endpoints_path: str):
    """Test creating a timer store from endpoint configuration."""
    store = read_endpoint_config(endpoints_path, endpoint_type="timer_store")

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        timer_store = RedisSessionTimerStore(
            config=RedisSessionTimerStoreConfig(
                host="localhost",
                port=6379,
                db=2,
                username="username",
                password="password",
                use_ssl=True,
                ssl_keyfile="keyfile.key",
                ssl_certfile="certfile.crt",
                ssl_ca_certs="my-bundle.ca-bundle",
                key_prefix="timer",
            ),
        )
        created_timer_store = SessionTimerStore.create(store)

    assert isinstance(timer_store, type(created_timer_store))


@pytest.mark.asyncio
async def test_redis_store_timer(redis_timer_store, mock_redis):
    """Redis store_timer uses sorted set and hash."""
    scheduled_time = time.time() + 10.0

    with capture_logs() as caplog:
        await redis_timer_store.store_timer(
            sender_id="test_sender",
            session_id="test_session",
            scheduled_time=scheduled_time,
            metadata={"key": "value"},
        )

    mock_redis.pipeline.assert_called_once()
    mock_redis.zadd.assert_called_once()
    mock_redis.hset.assert_called_once()
    mock_redis.execute.assert_called_once()
    logs = filter_logs(caplog, "timer_store.redis.timer_stored", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"
    assert logs[0]["session_id"] == "test_session"


@pytest.mark.asyncio
async def test_redis_delete_timer(redis_timer_store, mock_redis):
    """Redis delete_timer uses Lua script for unconditional delete."""
    redis_timer_store._delete_timer_script = Mock(return_value=1)

    with capture_logs() as caplog:
        result = await redis_timer_store.delete_timer("test_sender")

    assert result is True
    redis_timer_store._delete_timer_script.assert_called_once()
    call_kwargs = redis_timer_store._delete_timer_script.call_args
    assert call_kwargs.kwargs["args"][0] == "test_sender"
    assert call_kwargs.kwargs["args"][1] == ""  # Empty string = unconditional
    logs = filter_logs(caplog, "timer_store.redis.timer_deleted", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"


@pytest.mark.asyncio
async def test_redis_delete_timer_only_if_scheduled_time(redis_timer_store, mock_redis):
    """Redis delete_timer with only_if_scheduled_time uses Lua script for
    conditional atomic delete."""
    scheduled_time = 1234567890.5
    redis_timer_store._delete_timer_script = Mock(return_value=1)

    result = await redis_timer_store.delete_timer(
        "test_sender", only_if_scheduled_time=scheduled_time
    )

    assert result is True
    redis_timer_store._delete_timer_script.assert_called_once()
    call_kwargs = redis_timer_store._delete_timer_script.call_args
    assert call_kwargs.kwargs["args"][0] == "test_sender"
    assert call_kwargs.kwargs["args"][1] == str(scheduled_time)


@pytest.mark.asyncio
async def test_redis_get_timer(redis_timer_store, mock_redis):
    """Redis get_timer retrieves from hash."""
    mock_redis.hget.return_value = timer_json_bytes(
        "test_sender", "test_session", 1234567890.0
    )
    timer = await redis_timer_store.get_timer("test_sender")

    assert_timer(
        timer,
        sender_id="test_sender",
        session_id="test_session",
        scheduled_time=1234567890.0,
    )
    mock_redis.hget.assert_called_once()


@pytest.mark.asyncio
async def test_redis_get_timer_not_found(redis_timer_store, mock_redis):
    """Redis get_timer returns None when not found."""
    mock_redis.hget.return_value = None

    timer = await redis_timer_store.get_timer("nonexistent")

    assert timer is None


@pytest.mark.asyncio
async def test_redis_get_expired_timers(redis_timer_store, mock_redis):
    """Redis get_expired_timers uses ZRANGEBYSCORE."""
    mock_redis.zrangebyscore.return_value = [b"sender_1", b"sender_2"]
    mock_redis.hget.side_effect = [
        timer_json_bytes("sender_1", "s1", 100.0),
        timer_json_bytes("sender_2", "s2", 200.0),
    ]

    timers = await redis_timer_store.get_expired_timers(cutoff_time=300.0)

    assert len(timers) == 2
    assert timers[0].sender_id == "sender_1"
    assert timers[1].sender_id == "sender_2"
    mock_redis.zrangebyscore.assert_called_once()


@pytest.mark.asyncio
async def test_redis_get_expired_timers_empty(redis_timer_store, mock_redis):
    """Redis get_expired_timers returns empty list when no expired timers."""
    mock_redis.zrangebyscore.return_value = []

    timers = await redis_timer_store.get_expired_timers(cutoff_time=300.0)

    assert timers == []


def test_redis_timer_store_close(redis_timer_store, mock_redis):
    """Redis timer store closes connection."""
    redis_timer_store.close()

    mock_redis.close.assert_called_once()


def test_redis_timer_store_close_handles_error(redis_timer_store, mock_redis):
    """Redis timer store handles close errors gracefully."""
    mock_redis.close.side_effect = Exception("Connection error")

    redis_timer_store.close()
