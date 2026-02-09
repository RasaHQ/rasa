from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from rasa.core.agent import Agent
from rasa.core.redis_connection_factory import DeploymentMode
from rasa.core.timer_store import (
    InMemorySessionTimerStore,
    RedisSessionTimerStore,
    RedisSessionTimerStoreConfig,
    SessionTimerStore,
    _create_from_endpoint_config,
)
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config


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
  use_ssl: True
  ssl_keyfile: "keyfile.key"
  ssl_certfile: "certfile.crt"
  ssl_ca_certs: "my-bundle.ca-bundle"
"""
    )
    return str(endpoints_file)


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
        # Given
        mock_redis = Mock()
        mock_create.return_value = mock_redis

        base_config = {"host": "localhost", "port": 6379, "db": 2}
        config = RedisSessionTimerStoreConfig(**{**base_config, **extra_config})

        # When
        timer_store = RedisSessionTimerStore(config=config)

        # Then
        assert isinstance(timer_store, RedisSessionTimerStore)
        assert timer_store.red == mock_redis

        mock_create.assert_called_once()
        call_args = mock_create.call_args
        config = call_args.args[0]

        assert config.deployment_mode == expected_mode
        assert config.endpoints == expected_endpoints
        assert config.sentinel_service == expected_sentinel_service


def test_create_in_memory_timer_store():
    """Test creating an in-memory timer store."""
    timer_store = InMemorySessionTimerStore()
    assert isinstance(timer_store, InMemorySessionTimerStore)
    assert timer_store.timers == {}


@pytest.mark.parametrize(
    "endpoint_config",
    [
        None,
        EndpointConfig(type="in_memory"),
    ],
)
def test_create_in_memory_timer_store_from_config(endpoint_config):
    """Test that None or in_memory config creates an in-memory timer store."""
    timer_store = _create_from_endpoint_config(endpoint_config)
    assert isinstance(timer_store, InMemorySessionTimerStore)


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
    "timer_store_class,kwargs",
    [
        (InMemorySessionTimerStore, {}),
        (
            RedisSessionTimerStore,
            {"config": RedisSessionTimerStoreConfig(host="localhost", port=6379, db=2)},
        ),
    ],
)
def test_timer_store_propagated_to_agent(timer_store_class, kwargs):
    """Test that timer store is correctly propagated to Agent."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        timer_store = timer_store_class(**kwargs)
        agent = Agent(timer_store=timer_store)

        assert agent.timer_store is timer_store
        assert isinstance(agent.timer_store, timer_store_class)


def test_timer_store_created_via_factory_in_agent():
    """Test that timer store is created via factory when None is passed."""
    agent = Agent(timer_store=None)

    assert agent.timer_store is not None
    assert isinstance(agent.timer_store, InMemorySessionTimerStore)


@pytest.mark.parametrize(
    "timer_store_class,kwargs",
    [
        (InMemorySessionTimerStore, {}),
        (
            RedisSessionTimerStore,
            {"config": RedisSessionTimerStoreConfig(host="localhost", port=6379, db=2)},
        ),
    ],
)
def test_timer_store_propagated_to_processor(timer_store_class, kwargs):
    """Test that timer store is correctly propagated to MessageProcessor."""
    from rasa.core.lock_store import InMemoryLockStore
    from rasa.core.nlg import TemplatedNaturalLanguageGenerator
    from rasa.core.processor import MessageProcessor
    from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore

    # Create mock objects for the model loading return values
    mock_graph_runner = Mock()
    mock_graph_runner._graph_schema.nodes = {}
    mock_model_metadata = Mock()
    mock_model_metadata.assistant_id = "test_assistant"

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        timer_store = timer_store_class(**kwargs)

        with patch.object(
            MessageProcessor,
            "_load_model",
            return_value=("model.tar.gz", mock_model_metadata, mock_graph_runner),
        ):
            processor = MessageProcessor(
                model_path="dummy_path",
                tracker_store=InMemoryTrackerStore(domain=None),
                lock_store=InMemoryLockStore(),
                generator=TemplatedNaturalLanguageGenerator(responses={}),
                timer_store=timer_store,
            )

            assert processor.timer_store is timer_store
            assert isinstance(processor.timer_store, timer_store_class)
