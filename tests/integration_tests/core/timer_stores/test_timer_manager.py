from unittest.mock import AsyncMock, Mock, patch

import pytest

from rasa.core.agent import Agent
from rasa.core.lock_store import InMemoryLockStore
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.processor import MessageProcessor
from rasa.core.timer_managers.in_memory_timer_manager import InMemorySessionTimerManager
from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
from tests.core.timer_test_helpers import create_redis_manager_and_store


@pytest.fixture
def mock_redis() -> Mock:
    """Create a mock Redis client for timer manager integration tests."""
    mock = Mock()
    mock.pipeline.return_value = mock
    mock.execute.return_value = [1, 1]
    mock.eval.return_value = 1
    return mock


@pytest.mark.parametrize(
    "timer_manager_class,kwargs",
    [
        (InMemorySessionTimerManager, {}),
        (InMemorySessionTimerManager, {"store": None}),
    ],
)
def test_timer_manager_propagated_to_agent(timer_manager_class, kwargs):
    """Test that timer manager is correctly propagated to Agent."""
    timer_manager = timer_manager_class(**kwargs)
    agent = Agent(timer_manager=timer_manager)
    assert agent.timer_manager is timer_manager
    assert isinstance(agent.timer_manager, timer_manager_class)


@pytest.mark.parametrize(
    "timer_manager_class,kwargs",
    [
        (InMemorySessionTimerManager, {}),
        (InMemorySessionTimerManager, {"store": None}),
    ],
)
def test_timer_manager_propagated_to_processor(timer_manager_class, kwargs):
    """Test that timer manager is correctly propagated to MessageProcessor."""
    timer_manager = timer_manager_class(**kwargs)
    mock_graph_runner = Mock()
    mock_graph_runner._graph_schema.nodes = {}
    mock_model_metadata = Mock()
    mock_model_metadata.assistant_id = "test_assistant"

    with patch.object(
        MessageProcessor,
        "_load_model",
        return_value=(
            "model.tar.gz",
            mock_model_metadata,
            mock_graph_runner,
        ),
    ):
        processor = MessageProcessor(
            model_path="dummy_path",
            tracker_store=InMemoryTrackerStore(domain=None),
            lock_store=InMemoryLockStore(),
            generator=TemplatedNaturalLanguageGenerator(responses={}),
            timer_manager=timer_manager,
        )
    assert processor.timer_manager is timer_manager
    assert isinstance(processor.timer_manager, timer_manager_class)


@pytest.mark.asyncio
async def test_agent_initialize_timer_manager_sets_callback_and_starts():
    """Agent.initialize_timer_manager sets callback and starts manager."""
    timer_manager = InMemorySessionTimerManager()
    mock_graph_runner = Mock()
    mock_graph_runner._graph_schema.nodes = {}
    mock_model_metadata = Mock()
    mock_model_metadata.assistant_id = "test_assistant"
    mock_model_metadata.domain = None
    mock_model_metadata.training_type = None

    agent = Agent(timer_manager=timer_manager)

    with patch.object(
        MessageProcessor,
        "_load_model",
        return_value=(
            "model.tar.gz",
            mock_model_metadata,
            mock_graph_runner,
        ),
    ):
        agent.processor = MessageProcessor(
            model_path="dummy_path",
            tracker_store=InMemoryTrackerStore(domain=None),
            lock_store=InMemoryLockStore(),
            generator=TemplatedNaturalLanguageGenerator(responses={}),
            timer_manager=timer_manager,
        )

    await agent.initialize_timer_manager()

    # Verify manager is running
    assert timer_manager._running is True

    await agent.close()


@pytest.mark.asyncio
async def test_agent_close_stops_timer_manager():
    """Agent.close stops the timer manager."""
    timer_manager = InMemorySessionTimerManager()
    agent = Agent(timer_manager=timer_manager)

    # Start the manager manually
    await timer_manager.start()
    assert timer_manager._running is True

    await agent.close()

    # Manager should be stopped
    assert timer_manager._running is False


@pytest.mark.asyncio
async def test_redis_manager_set_callback_enables_start(mock_redis):
    """Redis manager can start after set_callback is called."""
    with patch(
        "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection",
        return_value=mock_redis,
    ):
        _, manager = create_redis_manager_and_store(mock_redis)

        # Should fail without callback
        with pytest.raises(RuntimeError, match="Must call set_callback"):
            await manager.start()

        # Set callback
        manager.set_callback(AsyncMock())

        # Now start should work
        mock_redis.zrangebyscore.return_value = []
        await manager.start()
        assert manager._running is True

        await manager.stop()
