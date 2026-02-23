from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Text

import structlog

from rasa.core.timer_managers.timer_manager import SessionTimerManager
from rasa.core.timer_stores.in_memory_timer_store import InMemorySessionTimerStore
from rasa.core.timer_stores.timer_store import SessionTimerStore, TimerCallback

structlogger = structlog.getLogger(__name__)


class InMemorySessionTimerManager(SessionTimerManager):
    """In-memory timer manager using asyncio.Task for scheduling."""

    def __init__(self, store: Optional[SessionTimerStore] = None) -> None:
        """Initialize the in-memory timer manager.

        Args:
            store: The underlying timer store. Creates one if not provided.
        """
        if store is None:
            store = InMemorySessionTimerStore()
        super().__init__(store)
        self._tasks: Dict[Text, asyncio.Task[None]] = {}
        self._locks: Dict[Text, asyncio.Lock] = {}
        self._running = False

    def set_callback(self, callback: TimerCallback) -> None:
        """No-op for in-memory manager which uses per-timer callbacks."""
        ...

    async def start(self) -> None:
        """Start the timer manager."""
        self._running = True
        structlogger.debug("timer_manager.in_memory.started")

    async def stop(self) -> None:
        """Stop the timer manager and cancel all pending timers."""
        self._running = False
        tasks_to_await = []

        # Cancel all tasks and delete their timers from the store.
        for sender_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                tasks_to_await.append(task)
            await self._store.delete_timer(sender_id)
        self._tasks.clear()

        # Properly await cancelled tasks to avoid "Task was destroyed" warnings.
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)

        self._store.close()
        structlogger.debug("timer_manager.in_memory.stopped")

    async def schedule_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        timeout_seconds: float,
        callback: TimerCallback,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Schedule a timer using asyncio.Task.

        Uses per-sender locking to prevent race conditions when multiple
        concurrent calls schedule timers for the same sender_id.
        """
        lock = self._get_lock(sender_id)
        async with lock:
            # Cancel existing timer if any
            await self._cancel_timer_unlocked(sender_id)

            scheduled_time = time.time() + timeout_seconds

            # Store timer data
            await self._store.store_timer(
                sender_id, session_id, scheduled_time, metadata
            )

            task = asyncio.create_task(
                self._run_timer_task(
                    sender_id, session_id, scheduled_time, timeout_seconds, callback
                )
            )
            task.add_done_callback(self._make_task_done_callback(sender_id, session_id))
            self._tasks[sender_id] = task

            structlogger.debug(
                "timer_manager.in_memory.timer_scheduled",
                sender_id=sender_id,
                session_id=session_id,
                timeout_seconds=timeout_seconds,
            )

    async def cancel_timer(self, sender_id: Text) -> bool:
        """Cancel a scheduled timer."""
        lock = self._get_lock(sender_id)
        async with lock:
            return await self._cancel_timer_unlocked(sender_id)

    def _get_lock(self, sender_id: Text) -> asyncio.Lock:
        """Get or create a lock for a sender_id."""
        if sender_id not in self._locks:
            self._locks[sender_id] = asyncio.Lock()
        return self._locks[sender_id]

    async def _run_timer_task(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        scheduled_time: float,
        timeout_seconds: float,
        callback: TimerCallback,
    ) -> None:
        """Wait for timeout and trigger callback if timer is still active.

        Uses the same per-sender lock as scheduling and cancellation to
        avoid races with concurrent reset / cancel operations.
        """
        try:
            await asyncio.sleep(timeout_seconds)

            # Acquire lock to safely check and modify timer state
            lock = self._get_lock(sender_id)
            async with lock:
                # Verify this task still corresponds to the active timer.
                # If the timer was reset/cancelled, another operation will
                # have stored a different scheduled_time or removed the timer.
                current_timer = await self._store.get_timer(sender_id)

                if (
                    current_timer is None
                    or current_timer.session_id != session_id
                    or current_timer.scheduled_time != scheduled_time
                ):
                    # Timer was reset or cancelled; this task is stale
                    return

                # This task still owns the active timer; atomic claim
                self._tasks.pop(sender_id, None)
                deleted = await self._store.delete_timer(
                    sender_id, only_if_scheduled_time=scheduled_time
                )
                if not deleted:
                    return

            # Invoke callback outside the lock to avoid blocking
            await callback(sender_id, session_id, scheduled_time)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            structlogger.error(
                "timer_manager.in_memory.callback_error",
                sender_id=sender_id,
                session_id=session_id,
                error=str(e),
            )

    def _make_task_done_callback(
        self, sender_id: Text, session_id: Optional[Text]
    ) -> Callable[[asyncio.Future[None]], None]:
        """Create a done callback for a timer task.

        Retrieves task exception to avoid 'CancelledError' warning and logs
        any other exceptions.
        """

        def _on_task_done(future: asyncio.Future[None]) -> None:
            try:
                exc = future.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                structlogger.warning(
                    "timer_manager.in_memory.task_exception",
                    sender_id=sender_id,
                    session_id=session_id,
                    error=str(exc),
                )

        return _on_task_done

    async def _cancel_timer_unlocked(self, sender_id: Text) -> bool:
        """Cancel a scheduled timer without acquiring lock.

        To be used within a locked context.
        """
        task = self._tasks.pop(sender_id, None)
        deleted = await self._store.delete_timer(sender_id)

        if task is not None and not task.done():
            task.cancel()
            structlogger.debug(
                "timer_manager.in_memory.timer_cancelled",
                sender_id=sender_id,
            )
            return True

        return deleted
