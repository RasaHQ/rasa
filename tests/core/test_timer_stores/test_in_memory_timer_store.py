import time

import pytest
from structlog.testing import capture_logs

from rasa.core.timer_stores.in_memory_timer_store import InMemorySessionTimerStore
from rasa.core.timer_stores.timer_store import (
    _create_store_from_endpoint_config,
)
from tests.core.timer_test_helpers import (
    IN_MEMORY_ENDPOINT_CONFIGS,
    assert_timer,
    store_test_timer,
)
from tests.utilities import filter_logs


@pytest.mark.asyncio
async def test_create_in_memory_timer_store():
    """Test creating an in-memory timer store."""
    timer_store = InMemorySessionTimerStore()
    assert isinstance(timer_store, InMemorySessionTimerStore)
    assert await timer_store.get_timer("nonexistent") is None


@pytest.mark.parametrize("endpoint_config", IN_MEMORY_ENDPOINT_CONFIGS)
def test_create_in_memory_timer_store_from_config(endpoint_config):
    """Test that None or in_memory config creates an in-memory timer store."""
    timer_store = _create_store_from_endpoint_config(endpoint_config)
    assert isinstance(timer_store, InMemorySessionTimerStore)


@pytest.fixture
def timer_store() -> InMemorySessionTimerStore:
    """Create a fresh in-memory timer store for each test."""
    return InMemorySessionTimerStore()


@pytest.mark.asyncio
async def test_store_stores_timer_data(timer_store: InMemorySessionTimerStore):
    """Timer data is stored correctly."""
    scheduled_time = time.time() + 10.0

    with capture_logs() as caplog:
        await timer_store.store_timer(
            sender_id="test_sender",
            session_id="test_session",
            scheduled_time=scheduled_time,
            metadata={"key": "value"},
        )
    logs = filter_logs(caplog, "timer_store.in_memory.timer_stored", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"
    assert logs[0]["session_id"] == "test_session"

    timer = await timer_store.get_timer("test_sender")
    assert_timer(
        timer,
        sender_id="test_sender",
        session_id="test_session",
        scheduled_time=scheduled_time,
        metadata={"key": "value"},
    )


@pytest.mark.asyncio
async def test_delete_timer_removes_data(timer_store: InMemorySessionTimerStore):
    """Timer data can be deleted."""
    await store_test_timer(timer_store)

    with capture_logs() as caplog:
        result = await timer_store.delete_timer("test_sender")

    assert result is True
    assert await timer_store.get_timer("test_sender") is None
    logs = filter_logs(caplog, "timer_store.in_memory.timer_deleted", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"


@pytest.mark.asyncio
async def test_delete_timer_returns_false_for_nonexistent(
    timer_store: InMemorySessionTimerStore,
):
    """Delete returns False when no timer exists."""
    result = await timer_store.delete_timer("nonexistent_sender")
    assert result is False


@pytest.mark.asyncio
async def test_delete_timer_only_if_scheduled_time_matches(
    timer_store: InMemorySessionTimerStore,
):
    """Delete with only_if_scheduled_time removes only when scheduled_time matches."""
    scheduled_time = time.time() + 60.0
    await timer_store.store_timer(
        sender_id="test_sender",
        session_id="session",
        scheduled_time=scheduled_time,
    )

    # Mismatch: does not delete
    result = await timer_store.delete_timer(
        "test_sender", only_if_scheduled_time=scheduled_time - 1.0
    )
    assert result is False
    assert await timer_store.get_timer("test_sender") is not None

    # Match: deletes
    result = await timer_store.delete_timer(
        "test_sender", only_if_scheduled_time=scheduled_time
    )
    assert result is True
    assert await timer_store.get_timer("test_sender") is None


@pytest.mark.asyncio
async def test_get_timer_returns_timer_data(timer_store: InMemorySessionTimerStore):
    """Get returns timer data when it exists."""
    await store_test_timer(timer_store)
    timer = await timer_store.get_timer("test_sender")

    assert_timer(
        timer,
        sender_id="test_sender",
        session_id="test_session",
    )


@pytest.mark.asyncio
async def test_get_timer_returns_none_for_nonexistent(
    timer_store: InMemorySessionTimerStore,
):
    """Get returns None when no timer exists."""
    timer = await timer_store.get_timer("nonexistent_sender")
    assert timer is None


@pytest.mark.asyncio
async def test_get_expired_timers_filters_by_time(
    timer_store: InMemorySessionTimerStore,
):
    """get_expired_timers returns timers scheduled before given time."""
    current_time = time.time()

    await timer_store.store_timer(
        sender_id="sender_1",
        session_id="session_1",
        scheduled_time=current_time + 5.0,
    )
    await timer_store.store_timer(
        sender_id="sender_2",
        session_id="session_2",
        scheduled_time=current_time + 15.0,
    )

    # At current_time + 10, only sender_1 should be expired
    expired = await timer_store.get_expired_timers(cutoff_time=current_time + 10.0)

    assert len(expired) == 1
    assert expired[0].sender_id == "sender_1"


@pytest.mark.asyncio
async def test_timer_store_close_clears_data(timer_store: InMemorySessionTimerStore):
    """Close clears all timer data."""
    await timer_store.store_timer(
        sender_id="test",
        session_id="session",
        scheduled_time=time.time() + 60.0,
    )

    with capture_logs() as caplog:
        timer_store.close()

    assert await timer_store.get_timer("test") is None
    logs = filter_logs(caplog, "timer_store.in_memory.closed", "debug")
    assert len(logs) == 1
