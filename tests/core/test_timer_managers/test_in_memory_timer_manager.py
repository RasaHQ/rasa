import asyncio
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from rasa.core.timer_managers.in_memory_timer_manager import InMemorySessionTimerManager
from rasa.core.timer_stores.timer_store import SessionTimer
from tests.core.timer_test_helpers import assert_timer
from tests.utilities import filter_logs


@pytest.fixture
async def timer_manager() -> AsyncIterator[InMemorySessionTimerManager]:
    """Create a fresh in-memory timer manager for each test."""
    manager = InMemorySessionTimerManager()
    yield manager
    await manager.stop()


@pytest.mark.asyncio
async def test_schedule_timer_creates_task(timer_manager: InMemorySessionTimerManager):
    """Timer is scheduled and stored."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=10.0,
        callback=callback,
    )
    timer = await timer_manager.get_timer("test_sender")
    assert_timer(
        timer,
        sender_id="test_sender",
        session_id="test_session",
    )


@pytest.mark.asyncio
async def test_timer_fires_callback_after_timeout(
    timer_manager: InMemorySessionTimerManager,
):
    """Callback is invoked after the timeout period."""
    callback = AsyncMock()
    with capture_logs() as caplog:
        with patch(
            "rasa.core.timer_managers.in_memory_timer_manager.time"
        ) as mock_time:
            mock_time.time.return_value = 1000.0
            await timer_manager.schedule_timer(
                sender_id="test_sender",
                session_id="test_session",
                timeout_seconds=0.05,
                callback=callback,
            )

    await asyncio.sleep(0.1)

    callback.assert_called_once_with("test_sender", "test_session", 1000.05)
    assert await timer_manager.get_timer("test_sender") is None
    logs = filter_logs(caplog, "timer_manager.in_memory.timer_scheduled", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"
    assert logs[0]["session_id"] == "test_session"


@pytest.mark.asyncio
async def test_cancel_timer_stops_pending_task(
    timer_manager: InMemorySessionTimerManager,
):
    """Timer can be cancelled before it fires."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=10.0,
        callback=callback,
    )

    with capture_logs() as caplog:
        result = await timer_manager.cancel_timer("test_sender")

    assert result is True
    assert await timer_manager.get_timer("test_sender") is None
    callback.assert_not_called()
    logs = filter_logs(caplog, "timer_manager.in_memory.timer_cancelled", "debug")
    assert len(logs) == 1
    assert logs[0]["sender_id"] == "test_sender"


@pytest.mark.asyncio
async def test_cancel_timer_returns_false_for_nonexistent(
    timer_manager: InMemorySessionTimerManager,
):
    """Cancel returns False when no timer exists for the sender."""
    result = await timer_manager.cancel_timer("nonexistent_sender")
    assert result is False


@pytest.mark.asyncio
async def test_schedule_timer_updates_session_and_time(
    timer_manager: InMemorySessionTimerManager,
):
    """Scheduling a timer updates session_id and scheduled_time."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="session_1",
        timeout_seconds=10.0,
        callback=callback,
    )

    old_timer = await timer_manager.get_timer("test_sender")
    assert old_timer is not None
    old_scheduled_time = old_timer.scheduled_time

    await asyncio.sleep(0.01)

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="session_2",
        timeout_seconds=10.0,
        callback=callback,
    )

    new_timer = await timer_manager.get_timer("test_sender")
    assert new_timer is not None
    assert new_timer.session_id == "session_2"
    assert new_timer.scheduled_time > old_scheduled_time


@pytest.mark.asyncio
async def test_close_cancels_all_timers(timer_manager: InMemorySessionTimerManager):
    """Close method cancels all pending timers."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="sender_1",
        session_id="session_1",
        timeout_seconds=10.0,
        callback=callback,
    )
    await timer_manager.schedule_timer(
        sender_id="sender_2",
        session_id="session_2",
        timeout_seconds=10.0,
        callback=callback,
    )

    await timer_manager.stop()

    assert await timer_manager.get_timer("sender_1") is None
    assert await timer_manager.get_timer("sender_2") is None


@pytest.mark.asyncio
async def test_get_timer_returns_timer_info(timer_manager: InMemorySessionTimerManager):
    """Get returns timer info when timer exists."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=10.0,
        callback=callback,
    )

    timer = await timer_manager.get_timer("test_sender")

    assert timer is not None
    assert isinstance(timer, SessionTimer)
    assert timer.sender_id == "test_sender"
    assert timer.session_id == "test_session"


@pytest.mark.asyncio
async def test_schedule_timer_with_metadata(timer_manager: InMemorySessionTimerManager):
    """Timer can be scheduled with metadata."""
    callback = AsyncMock()

    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="test_session",
        timeout_seconds=10.0,
        callback=callback,
        metadata={"reason": "session_timeout", "channel": "web"},
    )

    timer = await timer_manager.get_timer("test_sender")
    assert timer is not None
    assert timer.metadata == {"reason": "session_timeout", "channel": "web"}


@pytest.mark.asyncio
async def test_schedule_timer_replaces_existing(
    timer_manager: InMemorySessionTimerManager,
):
    """Scheduling a timer for same sender_id replaces existing timer."""
    callback1 = AsyncMock()
    callback2 = AsyncMock()

    with patch("rasa.core.timer_managers.in_memory_timer_manager.time") as mock_time:
        mock_time.time.return_value = 1000.0
        await timer_manager.schedule_timer(
            sender_id="test_sender",
            session_id="session_1",
            timeout_seconds=10.0,
            callback=callback1,
        )

        await timer_manager.schedule_timer(
            sender_id="test_sender",
            session_id="session_2",
            timeout_seconds=0.05,
            callback=callback2,
        )

    await asyncio.sleep(0.1)

    callback1.assert_not_called()
    callback2.assert_called_once_with("test_sender", "session_2", 1000.05)


@pytest.mark.asyncio
async def test_manager_start_stop(timer_manager: InMemorySessionTimerManager):
    """Manager start and stop lifecycle works correctly."""
    with capture_logs() as caplog:
        await timer_manager.start()
    assert timer_manager._running is True
    logs = filter_logs(caplog, "timer_manager.in_memory.started", "debug")
    assert len(logs) == 1

    with capture_logs() as caplog:
        await timer_manager.stop()
    assert timer_manager._running is False
    logs = filter_logs(caplog, "timer_manager.in_memory.stopped", "debug")
    assert len(logs) == 1


def test_manager_with_custom_store():
    """Manager can be initialized with a custom store."""
    store = MagicMock()
    manager = InMemorySessionTimerManager(store=store)

    assert manager.store is store


# ============================================================================
# Race Condition Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_timer_expiration_and_message(
    timer_manager: InMemorySessionTimerManager,
):
    """Timer fires while message resets timer - reset should win."""
    await timer_manager.start()

    async def slow_callback(*_: Any) -> None:
        await asyncio.sleep(0.05)  # Simulate slow callback processing

    # Schedule timer to fire quickly
    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="session_1",
        timeout_seconds=0.01,
        callback=slow_callback,
    )

    await asyncio.sleep(0.005)
    # Wait a tiny bit then reschedule before callback completes
    # Schedule new timer (simulates user message arriving)
    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="session_2",
        timeout_seconds=10.0,  # Long timeout
        callback=slow_callback,
    )

    # Wait for any pending callbacks
    await asyncio.sleep(0.1)

    # Timer should exist (reset succeeded)
    timer = await timer_manager.get_timer("test_sender")
    assert timer is not None
    assert timer.session_id == "session_2"

    await timer_manager.stop()


@pytest.mark.asyncio
async def test_multiple_rapid_schedules(timer_manager: InMemorySessionTimerManager):
    """Multiple rapid timer schedules should result in only one active timer."""
    await timer_manager.start()

    callback = AsyncMock()

    # Rapidly schedule timer multiple times for same sender
    for i in range(10):
        await timer_manager.schedule_timer(
            sender_id="test_sender",
            session_id=f"session_{i}",
            timeout_seconds=10.0,
            callback=callback,
        )

    timer = await timer_manager.get_timer("test_sender")

    assert timer is not None
    assert timer.session_id == "session_9"  # Last schedule wins
    assert len(timer_manager._tasks) == 1

    await timer_manager.stop()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_timer_fires_during_cancel(timer_manager: InMemorySessionTimerManager):
    """Timer callback completes even if cancel is called concurrently."""
    await timer_manager.start()

    callback_completed = asyncio.Event()

    async def tracked_callback(
        sender_id: str, session_id: str, scheduled_time: float | None = None
    ) -> None:
        await asyncio.sleep(0.02)  # Simulate work
        callback_completed.set()

    # Schedule timer to fire immediately
    await timer_manager.schedule_timer(
        sender_id="test_sender",
        session_id="session_1",
        timeout_seconds=0.01,
        callback=tracked_callback,
    )

    # Wait for timer to start firing
    await asyncio.sleep(0.015)

    # Try to cancel during callback execution
    await timer_manager.cancel_timer("test_sender")

    # Wait for callback to complete
    await asyncio.sleep(0.05)
    await timer_manager.stop()

    assert callback_completed.is_set()


@pytest.mark.asyncio
async def test_concurrent_schedule_same_sender(
    timer_manager: InMemorySessionTimerManager,
):
    """Concurrent schedule calls for same sender should not corrupt state."""
    await timer_manager.start()

    callback = AsyncMock()

    # Schedule multiple timers concurrently for same sender
    async def schedule_timer(session_num: int) -> None:
        await timer_manager.schedule_timer(
            sender_id="test_sender",
            session_id=f"session_{session_num}",
            timeout_seconds=10.0,
            callback=callback,
        )

    # Run concurrent schedules
    await asyncio.gather(*[schedule_timer(i) for i in range(5)])

    # Should have exactly one timer and one task
    timer = await timer_manager.get_timer("test_sender")
    assert timer is not None
    assert len(timer_manager._tasks) == 1

    await timer_manager.stop()
    await asyncio.sleep(0)
