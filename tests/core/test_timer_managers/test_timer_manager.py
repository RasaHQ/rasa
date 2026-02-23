from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.core.timer_managers.in_memory_timer_manager import InMemorySessionTimerManager
from rasa.core.timer_managers.redis_timer_manager import RedisSessionTimerManager
from rasa.core.timer_managers.timer_manager import create_timer_manager
from rasa.utils.endpoints import EndpointConfig
from tests.core.timer_test_helpers import IN_MEMORY_ENDPOINT_CONFIGS


@pytest.mark.asyncio
async def test_create_in_memory_timer_manager():
    """Test creating an in-memory timer manager."""
    manager = InMemorySessionTimerManager()
    assert isinstance(manager, InMemorySessionTimerManager)
    assert await manager.get_timer("nonexistent") is None


@pytest.mark.parametrize("endpoint_config", IN_MEMORY_ENDPOINT_CONFIGS)
def test_create_timer_manager_in_memory(endpoint_config):
    """create_timer_manager returns InMemorySessionTimerManager."""
    manager = create_timer_manager(endpoint_config)
    assert isinstance(manager, InMemorySessionTimerManager)


def test_create_timer_manager_redis():
    """create_timer_manager returns RedisSessionTimerManager for redis."""
    callback = AsyncMock()

    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
    ):
        config = EndpointConfig(type="redis", url="localhost", port=6379, db=2)
        manager = create_timer_manager(config, callback=callback)

        assert isinstance(manager, RedisSessionTimerManager)


def test_create_timer_manager_custom_store():
    """create_timer_manager wraps a custom store in InMemorySessionTimerManager."""
    custom_store = MagicMock()

    with patch(
        "rasa.core.timer_managers.timer_manager._create_store_from_endpoint_config",
        return_value=custom_store,
    ):
        config = EndpointConfig(type="my.custom.TimerStore")
        manager = create_timer_manager(config)

    assert isinstance(manager, InMemorySessionTimerManager)
    assert manager.store is custom_store
