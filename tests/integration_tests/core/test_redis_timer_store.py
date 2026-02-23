import asyncio
import os
import time
from typing import Iterator
from unittest.mock import AsyncMock

import pytest

from rasa.core.timer_managers.redis_timer_manager import RedisSessionTimerManager
from rasa.core.timer_stores.redis_timer_store import (
    RedisSessionTimerStore,
    RedisSessionTimerStoreConfig,
)
from tests.core.timer_test_helpers import (
    assert_timer,
    store_test_timer,
    track_callback_factory,
)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))


@pytest.fixture(
    params=[
        {"deployment_mode": "standard"},
        {
            "deployment_mode": "cluster",
            "endpoints": [
                f"{REDIS_HOST}:7000",
                f"{REDIS_HOST}:7001",
                f"{REDIS_HOST}:7002",
            ],
        },
        {
            "deployment_mode": "sentinel",
            "endpoints": [
                f"{REDIS_HOST}:26379",
                f"{REDIS_HOST}:26380",
                f"{REDIS_HOST}:26381",
            ],
            "sentinel_service": "mymaster",
        },
    ]
)
def redis_timer_store(
    request: pytest.FixtureRequest,
) -> Iterator[RedisSessionTimerStore]:
    """Create a Redis timer store for testing across all deployment modes."""
    # we need one redis database per worker, otherwise
    # tests conflicts with each others when databases are flushed
    pytest_worker_id = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    redis_database = int(pytest_worker_id.replace("gw", ""))

    # Base configuration
    config_dict = {"host": REDIS_HOST, "port": REDIS_PORT, "key_prefix": "test"}

    # For cluster mode, don't set db (clusters only support db 0)
    if request.param["deployment_mode"] != "cluster":
        config_dict["db"] = redis_database

    config_dict.update(request.param)

    config = RedisSessionTimerStoreConfig(**config_dict)
    store = RedisSessionTimerStore(config=config)
    try:
        yield store
    finally:
        store.red.flushdb()
        store.close()


# ============================================================================
# Redis Timer Store Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_redis_store_timer_roundtrip(redis_timer_store: RedisSessionTimerStore):
    """Timer can be stored and retrieved from Redis."""
    scheduled_time = time.time() + 60.0

    await redis_timer_store.store_timer(
        sender_id="test_sender",
        session_id="test_session",
        scheduled_time=scheduled_time,
        metadata={"reason": "timeout"},
    )
    timer = await redis_timer_store.get_timer("test_sender")

    assert_timer(
        timer,
        sender_id="test_sender",
        session_id="test_session",
        scheduled_time=scheduled_time,
        metadata={"reason": "timeout"},
    )


@pytest.mark.asyncio
async def test_redis_delete_timer(redis_timer_store: RedisSessionTimerStore):
    """Timer can be deleted from Redis."""
    await store_test_timer(redis_timer_store)

    result = await redis_timer_store.delete_timer("test_sender")
    assert result is True

    assert await redis_timer_store.get_timer("test_sender") is None


@pytest.mark.asyncio
async def test_redis_get_expired_timers(redis_timer_store: RedisSessionTimerStore):
    """Expired timers can be queried from Redis."""
    current_time = time.time()

    # Store expired timer
    await redis_timer_store.store_timer(
        sender_id="expired_sender",
        session_id="s1",
        scheduled_time=current_time - 10.0,  # Already expired
    )

    # Store future timer
    await redis_timer_store.store_timer(
        sender_id="future_sender",
        session_id="s2",
        scheduled_time=current_time + 60.0,  # Not expired
    )

    expired = await redis_timer_store.get_expired_timers()

    assert len(expired) == 1
    assert expired[0].sender_id == "expired_sender"


@pytest.mark.asyncio
async def test_redis_timer_overwrite(redis_timer_store: RedisSessionTimerStore):
    """Storing a timer with same sender_id overwrites the existing one."""
    await redis_timer_store.store_timer(
        sender_id="test_sender",
        session_id="session_1",
        scheduled_time=time.time() + 30.0,
    )

    await redis_timer_store.store_timer(
        sender_id="test_sender",
        session_id="session_2",
        scheduled_time=time.time() + 60.0,
    )

    timer = await redis_timer_store.get_timer("test_sender")
    assert timer.session_id == "session_2"


# ============================================================================
# Redis Timer Manager Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_redis_manager_recovery_with_real_redis(
    redis_timer_store: RedisSessionTimerStore,
):
    """Manager recovers expired timers on startup with real Redis."""
    # Pre-populate Redis with expired timers
    current_time = time.time()
    await redis_timer_store.store_timer(
        sender_id="expired_1",
        session_id="s1",
        scheduled_time=current_time - 10.0,
    )
    await redis_timer_store.store_timer(
        sender_id="expired_2",
        session_id="s2",
        scheduled_time=current_time - 5.0,
    )

    # Create manager and track callbacks
    callback_senders, track_callback = track_callback_factory()

    manager = RedisSessionTimerManager(redis_timer_store, poll_interval=10.0)
    manager.set_callback(track_callback)

    # Start should trigger recovery
    await manager.start()
    await asyncio.sleep(0.1)

    # Both expired timers should have been processed
    assert "expired_1" in callback_senders
    assert "expired_2" in callback_senders

    # Timers should be deleted
    assert await redis_timer_store.get_timer("expired_1") is None
    assert await redis_timer_store.get_timer("expired_2") is None

    await manager.stop()


@pytest.mark.asyncio
async def test_redis_manager_polling_with_real_redis(
    redis_timer_store: RedisSessionTimerStore,
):
    """Manager polls for and processes expired timers."""
    callback_senders, track_callback = track_callback_factory()

    manager = RedisSessionTimerManager(redis_timer_store, poll_interval=0.05)
    manager.set_callback(track_callback)

    await manager.start()

    # Schedule a timer that expires quickly
    await manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=0.1,
        callback=track_callback,
    )

    # Wait for timer to expire and be processed
    await asyncio.sleep(0.3)

    assert "test_sender" in callback_senders
    assert await redis_timer_store.get_timer("test_sender") is None

    await manager.stop()


# ============================================================================
# Multi-Instance Race Condition Tests
# ============================================================================


@pytest.mark.asyncio
async def test_multiple_managers_process_different_timers(
    redis_timer_store: RedisSessionTimerStore,
):
    """Multiple manager instances don't reprocess the same timer.

    This simulates multiple pods processing timers from the same Redis.
    """
    senders_1 = []
    senders_2 = []

    async def slow_track_1(
        sender_id: str, session_id: str, scheduled_time: float | None = None
    ) -> None:
        await asyncio.sleep(0.01)
        senders_1.append(sender_id)

    async def slow_track_2(
        sender_id: str, session_id: str, scheduled_time: float | None = None
    ) -> None:
        await asyncio.sleep(0.01)
        senders_2.append(sender_id)

    # Create two managers sharing the same store
    manager1 = RedisSessionTimerManager(redis_timer_store, poll_interval=0.05)
    manager1.set_callback(slow_track_1)

    # Need a separate store instance for manager2 but same Redis connection
    config = redis_timer_store.config
    store2 = RedisSessionTimerStore(config=config)
    manager2 = RedisSessionTimerManager(store2, poll_interval=0.05)
    manager2.set_callback(slow_track_2)

    # Pre-populate with expired timers
    for i in range(5):
        await redis_timer_store.store_timer(
            sender_id=f"sender_{i}",
            session_id=f"session_{i}",
            scheduled_time=time.time() - 1.0,
        )

    # Start both managers
    await manager1.start()
    await manager2.start()

    # Wait for processing
    await asyncio.sleep(0.3)

    # Each timer should only be processed once total
    all_processed = set(senders_1 + senders_2)
    assert len(all_processed) == 5

    # No timer should be in both lists (no double processing)
    duplicates = set(senders_1) & set(senders_2)
    assert len(duplicates) == 0, f"Timers processed by both: {duplicates}"

    await manager1.stop()
    await manager2.stop()
    store2.close()


@pytest.mark.asyncio
async def test_concurrent_schedule_and_cancel(
    redis_timer_store: RedisSessionTimerStore,
):
    """Concurrent schedule and cancel operations don't corrupt state."""
    manager = RedisSessionTimerManager(redis_timer_store, poll_interval=10.0)
    manager.set_callback(AsyncMock())

    await manager.start()

    try:
        # Concurrent operations for same sender
        async def schedule() -> None:
            await manager.schedule_timer(
                sender_id="test_sender",
                session_id="session",
                timeout_seconds=60.0,
                callback=AsyncMock(),
            )

        async def cancel() -> None:
            await manager.cancel_timer("test_sender")

        # Run many concurrent operations
        tasks = [schedule() if i % 2 == 0 else cancel() for i in range(20)]
        await asyncio.gather(*tasks)

        # State should be consistent - either timer exists or not
        timer = await redis_timer_store.get_timer("test_sender")
        # Timer might exist or not depending on last operation
        assert timer is None or timer.sender_id == "test_sender"

    finally:
        await manager.stop()


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.asyncio
async def test_high_volume_timer_operations(
    redis_timer_store: RedisSessionTimerStore,
):
    """Store can handle high volume of timer operations."""
    num_timers = 100
    scheduled_time = time.time() + 60.0

    # Store many timers
    start = time.time()
    for i in range(num_timers):
        await redis_timer_store.store_timer(
            sender_id=f"sender_{i}",
            session_id=f"session_{i}",
            scheduled_time=scheduled_time,
        )
    store_duration = time.time() - start

    # Verify some timers exist
    timer = await redis_timer_store.get_timer("sender_0")
    assert timer is not None
    timer = await redis_timer_store.get_timer("sender_99")
    assert timer is not None

    # Delete all
    start = time.time()
    for i in range(num_timers):
        await redis_timer_store.delete_timer(f"sender_{i}")
    delete_duration = time.time() - start

    # Verify deleted
    timer = await redis_timer_store.get_timer("sender_0")
    assert timer is None

    # Performance assertions (generous limits for CI)
    assert store_duration < 10.0, f"Store took {store_duration:.2f}s"
    assert delete_duration < 10.0, f"Delete took {delete_duration:.2f}s"


@pytest.mark.asyncio
async def test_rapid_timer_resets(redis_timer_store: RedisSessionTimerStore):
    """Manager handles rapid timer resets efficiently."""
    manager = RedisSessionTimerManager(redis_timer_store, poll_interval=10.0)
    manager.set_callback(AsyncMock())

    await manager.start()

    try:
        # Rapidly schedule timer for same sender
        num_schedules = 50
        start = time.time()
        for i in range(num_schedules):
            await manager.schedule_timer(
                sender_id="test_sender",
                session_id=f"session_{i}",
                timeout_seconds=60.0,
                callback=AsyncMock(),
            )
        duration = time.time() - start

        # Should only have one timer with the latest session
        timer = await redis_timer_store.get_timer("test_sender")
        assert timer is not None
        assert timer.session_id == f"session_{num_schedules - 1}"

        # Should complete reasonably quickly
        assert duration < 10.0, f"Rapid schedules took {duration:.2f}s"

    finally:
        await manager.stop()
