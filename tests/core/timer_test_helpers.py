import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Text, Tuple

from rasa.utils.endpoints import EndpointConfig

IN_MEMORY_ENDPOINT_CONFIGS = [None, EndpointConfig(type="in_memory")]


def timer_json(
    sender_id: Text,
    session_id: Optional[Text],
    scheduled_time: float,
    metadata: Optional[Dict[Text, Any]] = None,
) -> str:
    """Build JSON string for a SessionTimer (for Redis mock responses)."""
    data = {
        "sender_id": sender_id,
        "session_id": session_id,
        "scheduled_time": scheduled_time,
        "metadata": metadata or {},
    }
    return json.dumps(data)


def timer_json_bytes(
    sender_id: Text,
    session_id: Optional[Text],
    scheduled_time: float,
    metadata: Optional[Dict[Text, Any]] = None,
) -> bytes:
    """Build bytes JSON for Redis mock hget return values."""
    return timer_json(sender_id, session_id, scheduled_time, metadata).encode()


def assert_timer(
    timer: Optional[Any],
    *,
    sender_id: Optional[Text] = None,
    session_id: Optional[Text] = None,
    scheduled_time: Optional[float] = None,
    metadata: Optional[Dict[Text, Any]] = None,
) -> None:
    """Assert timer exists and optional fields match."""
    assert timer is not None
    if sender_id is not None:
        assert timer.sender_id == sender_id
    if session_id is not None:
        assert timer.session_id == session_id
    if scheduled_time is not None:
        assert timer.scheduled_time == scheduled_time
    if metadata is not None:
        assert timer.metadata == metadata


async def schedule_test_timer(
    manager: Any,
    *,
    sender_id: Text = "test_sender",
    session_id: Text = "test_session",
    timeout_seconds: float = 10.0,
    callback: Optional[Any] = None,
    metadata: Optional[Dict[Text, Any]] = None,
) -> None:
    """Schedule a timer on a manager with common test defaults."""
    if callback is None:
        from unittest.mock import AsyncMock

        callback = AsyncMock()
    await manager.schedule_timer(
        sender_id=sender_id,
        session_id=session_id,
        timeout_seconds=timeout_seconds,
        callback=callback,
        metadata=metadata,
    )


async def store_test_timer(
    store: Any,
    *,
    sender_id: Text = "test_sender",
    session_id: Text = "test_session",
    scheduled_time: Optional[float] = None,
    metadata: Optional[Dict[Text, Any]] = None,
) -> None:
    """Store a timer on a store with common test defaults."""
    if scheduled_time is None:
        scheduled_time = time.time() + 60.0
    await store.store_timer(
        sender_id=sender_id,
        session_id=session_id,
        scheduled_time=scheduled_time,
        metadata=metadata,
    )


def track_callback_factory() -> Tuple[List[Text], Any]:
    """Return (senders_list, async_track_callback) for integration tests."""
    senders: List[Text] = []

    async def track_callback(
        sender_id: Text, session_id: Text, scheduled_time: Optional[float] = None
    ) -> None:
        senders.append(sender_id)

    return senders, track_callback


@asynccontextmanager
async def running_manager(manager: Any) -> AsyncIterator[Any]:
    """Async context manager that starts manager, yields, then stops."""
    await manager.start()
    try:
        yield manager
    finally:
        await manager.stop()


def create_redis_manager_and_store(
    mock_redis: Any,
    *,
    poll_interval: float = 0.05,
) -> Tuple[Any, Any]:
    """Create RedisSessionTimerStore and RedisSessionTimerManager for tests.

    Call inside a patch of RedisConnectionFactory.create_connection
    with return_value=mock_redis. Returns (store, manager).
    """
    from rasa.core.timer_managers.redis_timer_manager import RedisSessionTimerManager
    from rasa.core.timer_stores.redis_timer_store import (
        RedisSessionTimerStore,
        RedisSessionTimerStoreConfig,
    )

    config = RedisSessionTimerStoreConfig(
        host="localhost",
        port=6379,
        db=2,
        poll_interval=poll_interval,
    )
    store = RedisSessionTimerStore(config=config)
    manager = RedisSessionTimerManager(store, poll_interval=poll_interval)
    return store, manager
