import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from structlog.testing import capture_logs

from tests.core.timer_test_helpers import (
    create_redis_manager_and_store,
    schedule_test_timer,
    timer_json_bytes,
)
from tests.utilities import filter_logs

REDIS_CONNECTION_PATCH = (
    "rasa.core.redis_connection_factory.RedisConnectionFactory.create_connection"
)


@pytest.fixture
def redis_manager_and_store(mock_redis):
    """Factory: returns (store, manager) under patched Redis."""

    def _create(poll_interval=0.05):
        with patch(REDIS_CONNECTION_PATCH, return_value=mock_redis):
            return create_redis_manager_and_store(
                mock_redis,
                poll_interval=poll_interval,
            )

    return _create


@pytest.mark.asyncio
async def test_redis_manager_requires_callback(redis_manager_and_store):
    """Redis manager requires callback before starting."""
    _, manager = redis_manager_and_store()

    with pytest.raises(RuntimeError, match="Must call set_callback"):
        await manager.start()


@pytest.mark.asyncio
async def test_redis_manager_processes_expired_timers(
    mock_redis, redis_manager_and_store
):
    """Redis manager processes expired timers via polling."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=0.05)
    manager.set_callback(callback)

    mock_redis.zrangebyscore.return_value = [b"test_sender"]
    mock_redis.hget.return_value = timer_json_bytes(
        "test_sender", "test_session", 100.0
    )
    mock_redis.zscore.return_value = 100.0
    mock_redis.execute.return_value = [1, 1]

    with capture_logs() as caplog:
        await manager.start()
        await asyncio.sleep(0.1)
        await manager.stop()

    callback.assert_called_with("test_sender", "test_session", 100.0)
    logs = filter_logs(caplog, "timer_manager.redis.started", "debug")
    assert len(logs) == 1
    logs = filter_logs(caplog, "timer_manager.redis.stopped", "debug")
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_redis_manager_schedule_timer(redis_manager_and_store):
    """Redis manager schedules timer by storing in Redis."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store()
    manager.set_callback(callback)

    with capture_logs() as caplog:
        await schedule_test_timer(manager, callback=callback)

    mock_redis = manager.store.red
    mock_redis.pipeline.assert_called()
    mock_redis.zadd.assert_called()
    mock_redis.hset.assert_called()
    logs = filter_logs(caplog, "timer_manager.redis.timer_scheduled", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"


@pytest.mark.asyncio
async def test_redis_manager_cancel_timer(mock_redis, redis_manager_and_store):
    """Redis manager cancels timer by deleting from Redis via Lua script."""
    _, manager = redis_manager_and_store()

    with capture_logs() as caplog:
        result = await manager.cancel_timer("test_sender")

    assert result is True
    mock_redis.register_script.return_value.assert_called()
    logs = filter_logs(caplog, "timer_manager.redis.timer_cancelled", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"


@pytest.mark.asyncio
async def test_redis_manager_stop_cancels_poll_task(
    mock_redis, redis_manager_and_store
):
    """Redis manager stop cancels the poll task."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=0.05)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.return_value = []

    await manager.start()
    assert manager._running is True
    assert manager._poll_task is not None

    await manager.stop()
    assert manager._running is False


@pytest.mark.asyncio
async def test_redis_manager_handles_poll_errors(mock_redis, redis_manager_and_store):
    """Redis manager handles errors during polling gracefully."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=0.05)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.side_effect = [
        Exception("Connection error"),
        [],
    ]

    await manager.start()
    await asyncio.sleep(0.15)
    await manager.stop()

    assert mock_redis.zrangebyscore.call_count >= 2


# ============================================================================
# Redis Timer Recovery Tests (with mocked Redis)
# ============================================================================


@pytest.mark.asyncio
async def test_redis_manager_recovery_on_startup(mock_redis, redis_manager_and_store):
    """Redis manager processes expired timers immediately on startup."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=10.0)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.return_value = [
        b"expired_sender1",
        b"expired_sender2",
    ]
    mock_redis.hget.side_effect = [
        timer_json_bytes("expired_sender1", "s1", 100.0),
        timer_json_bytes("expired_sender2", "s2", 200.0),
    ]
    mock_redis.zscore.side_effect = [100.0, 200.0]
    mock_redis.execute.return_value = [1, 1]

    with capture_logs() as caplog:
        await manager.start()

        # Give recovery a moment to complete
        await asyncio.sleep(0.05)

        # Callback should have been called for both expired timers
        assert callback.call_count == 2
        callback.assert_any_call("expired_sender1", "s1", 100.0)
        callback.assert_any_call("expired_sender2", "s2", 200.0)

        await manager.stop()

    logs = filter_logs(caplog, "timer_manager.redis.recovery_started", "info")
    assert len(logs) == 1
    assert logs[0]["expired_count"] == 2
    logs = filter_logs(caplog, "timer_manager.redis.recovery_completed", "info")
    assert len(logs) == 1
    assert logs[0]["processed_count"] == 2


@pytest.mark.asyncio
async def test_redis_manager_recovery_no_expired_timers(
    mock_redis, redis_manager_and_store
):
    """Redis manager handles case with no expired timers on startup."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=10.0)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.return_value = []

    with capture_logs() as caplog:
        await manager.start()
        await asyncio.sleep(0.05)

        callback.assert_not_called()

        await manager.stop()

    logs = filter_logs(
        caplog, "timer_manager.redis.recovery_no_expired_timers", "debug"
    )
    assert len(logs) == 1


@pytest.mark.asyncio
async def test_redis_manager_recovery_handles_errors(
    mock_redis, redis_manager_and_store
):
    """Redis manager handles errors during recovery gracefully."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=10.0)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.side_effect = Exception("Redis connection error")

    # Should not raise - error is logged and manager continues
    await manager.start()
    await asyncio.sleep(0.05)
    await manager.stop()


@pytest.mark.asyncio
async def test_redis_manager_recovery_callback_error(
    mock_redis, redis_manager_and_store
):
    """Redis manager continues processing when one callback fails during recovery."""
    successful_calls = []

    async def sometimes_failing_callback(
        sender_id: str, session_id: str, scheduled_time: float | None = None
    ) -> None:
        if sender_id == "fail_sender":
            raise Exception("Callback failed")
        successful_calls.append(sender_id)

    _, manager = redis_manager_and_store(poll_interval=10.0)
    manager.set_callback(sometimes_failing_callback)
    mock_redis.zrangebyscore.return_value = [
        b"fail_sender",
        b"success_sender",
    ]
    mock_redis.hget.side_effect = [
        timer_json_bytes("fail_sender", "s1", 100.0),
        timer_json_bytes("success_sender", "s2", 200.0),
    ]
    mock_redis.zscore.side_effect = [100.0, 200.0]
    mock_redis.execute.return_value = [1, 1]

    await manager.start()
    await asyncio.sleep(0.05)

    # Only successful callback should be recorded
    assert "success_sender" in successful_calls
    assert "fail_sender" not in successful_calls

    await manager.stop()


@pytest.mark.asyncio
async def test_redis_manager_retry_on_schedule_failure(
    mock_redis, redis_manager_and_store
):
    """Redis manager retries schedule_timer on transient errors."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store()
    manager.set_callback(callback)
    mock_redis.execute.side_effect = [
        Exception("Connection error"),
        Exception("Connection error"),
        [1, 1],
        [1, 1],
    ]

    await schedule_test_timer(manager, callback=callback)

    assert mock_redis.execute.call_count >= 2


@pytest.mark.asyncio
async def test_redis_manager_retry_on_cancel_failure(
    mock_redis, redis_manager_and_store
):
    """Redis manager retries cancel_timer on transient errors."""
    _, manager = redis_manager_and_store()
    mock_redis.register_script.return_value.side_effect = [
        Exception("Connection error"),
        1,
    ]

    await manager.cancel_timer("test_sender")

    assert mock_redis.register_script.return_value.call_count >= 2


# ============================================================================
# Connection Resilience Tests
# ============================================================================


@pytest.mark.asyncio
async def test_redis_manager_logs_restored_connection(
    mock_redis, redis_manager_and_store
):
    """Redis manager logs when connection is restored after errors."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=0.02)
    manager.set_callback(callback)
    mock_redis.zrangebyscore.side_effect = [
        Exception("Connection error"),
        Exception("Connection error"),
        [],
        [],
        [],
    ]

    await manager.start()
    await asyncio.sleep(0.15)
    await manager.stop()

    # Should have called zrangebyscore multiple times
    assert mock_redis.zrangebyscore.call_count >= 3


# ============================================================================
# Fallback In-Memory Store Tests
# ============================================================================


@pytest.mark.asyncio
async def test_redis_manager_fallback_on_schedule_failure(
    mock_redis, redis_manager_and_store
):
    """Redis manager falls back to in-memory when schedule fails."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store()
    manager.set_callback(callback)
    # All retries fail
    mock_redis.execute.side_effect = Exception("Connection error")

    await schedule_test_timer(manager, timeout_seconds=60.0, callback=callback)

    # Should have activated fallback
    assert manager._using_fallback is True
    assert manager._fallback_store is not None

    # Timer should be in fallback store
    timer = await manager._fallback_store.get_timer("test_sender")
    assert timer is not None
    assert timer.sender_id == "test_sender"


@pytest.mark.asyncio
async def test_redis_manager_cancel_checks_fallback(
    mock_redis, redis_manager_and_store
):
    """Redis manager checks fallback store when cancelling."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store()
    manager.set_callback(callback)
    # Schedule fails, falls back to in-memory
    mock_redis.execute.side_effect = Exception("Connection error")
    await schedule_test_timer(manager, timeout_seconds=60.0, callback=callback)

    # Cancel should succeed from fallback
    result = await manager.cancel_timer("test_sender")
    assert result is True

    # Timer should be gone from fallback
    timer = await manager._fallback_store.get_timer("test_sender")
    assert timer is None


@pytest.mark.asyncio
async def test_redis_manager_processes_fallback_expired_timers(
    mock_redis, redis_manager_and_store
):
    """Redis manager processes expired timers from fallback store."""
    callback = AsyncMock()
    _, manager = redis_manager_and_store(poll_interval=0.05)
    manager.set_callback(callback)

    # First call fails (triggers fallback), then succeeds
    call_count = [0]

    def execute_side_effect(*args, **kwargs):
        call_count[0] += 1
        # Fail first 6 calls (2 operations x 3 retries)
        if call_count[0] <= 6:
            raise Exception("Connection error")
        return [1, 1]

    mock_redis.execute.side_effect = execute_side_effect
    mock_redis.zrangebyscore.return_value = []

    # Schedule a timer that will go to fallback
    await manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=0.01,  # Very short timeout
        callback=callback,
    )

    # Start polling
    await manager.start()
    await asyncio.sleep(0.15)
    await manager.stop()

    # Callback should have been called from fallback processing
    callback.assert_called()
