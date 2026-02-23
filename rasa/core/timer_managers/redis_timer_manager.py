from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any, Dict, Optional, Text

import structlog

from rasa.core.timer_managers.timer_manager import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_RETRY_BACKOFF_BASE,
    SessionTimerManager,
)
from rasa.core.timer_stores.in_memory_timer_store import InMemorySessionTimerStore
from rasa.core.timer_stores.redis_timer_store import RedisSessionTimerStore
from rasa.core.timer_stores.timer_store import (
    SessionTimer,
    SessionTimerStore,
    TimerCallback,
)

structlogger = structlog.getLogger(__name__)


class RedisSessionTimerManager(SessionTimerManager):
    """Redis-based timer manager using polling for expired timers.

    This manager persists timers in Redis, allowing them to survive process restarts.
    On startup, it recovers any timers that expired during downtime and processes
    them immediately.

    If Redis becomes unavailable, the manager falls back to an in-memory store
    to ensure timer scheduling continues. Timers scheduled during Redis outages
    will be lost if the process restarts before Redis recovers.
    """

    def __init__(
        self,
        store: RedisSessionTimerStore,
        poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ) -> None:
        """Initialize the Redis timer manager.

        Args:
            store: The underlying Redis timer store.
            poll_interval: Interval in seconds between polling for expired timers.
        """
        super().__init__(store)
        self._poll_interval = poll_interval
        self._poll_task: Optional[asyncio.Task[None]] = None
        self._callback: Optional[TimerCallback] = None
        self._running = False
        self._fallback_store: InMemorySessionTimerStore = InMemorySessionTimerStore()
        self._using_fallback = False

    def set_callback(self, callback: TimerCallback) -> None:
        """Set the callback to invoke when timers expire.

        For Redis manager, a single callback handles all timer expirations.
        The callback receives sender_id and session_id.
        """
        self._callback = callback

    async def start(self) -> None:
        """Start the polling loop for expired timers.

        On startup, this method first recovers any timers that expired during
        downtime (e.g., if the process was restarted). These expired timers
        are processed immediately before starting the regular polling loop.
        """
        if self._callback is None:
            raise RuntimeError(
                "Must call set_callback() before starting RedisSessionTimerManager"
            )

        self._running = True

        # Recover timers that expired during downtime
        await self._recover_expired_timers()

        self._poll_task = asyncio.create_task(self._poll_loop())
        structlogger.debug(
            "timer_manager.redis.started",
            poll_interval=self._poll_interval,
        )

    async def stop(self) -> None:
        """Stop the polling loop and clean up resources."""
        self._running = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
        self._poll_task = None

        self._fallback_store.close()
        self._using_fallback = False

        self._store.close()
        structlogger.debug("timer_manager.redis.stopped")

    async def schedule_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        timeout_seconds: float,
        callback: TimerCallback,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Schedule a timer by storing it in Redis.

        Note: For Redis manager, the callback parameter is stored but the
        manager's global callback (set via set_callback) is what actually
        gets invoked when the timer expires.

        This method includes retry logic for transient Redis errors.
        """
        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                # Delete existing timer if any
                await self._store.delete_timer(sender_id)

                scheduled_time = time.time() + timeout_seconds
                await self._store.store_timer(
                    sender_id, session_id, scheduled_time, metadata
                )

                structlogger.debug(
                    "timer_manager.redis.timer_scheduled",
                    sender_id=sender_id,
                    session_id=session_id,
                    timeout_seconds=timeout_seconds,
                )
                return
            except Exception as e:
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    structlogger.warning(
                        "timer_manager.redis.schedule_retry",
                        sender_id=sender_id,
                        attempt=attempt + 1,
                        max_retries=DEFAULT_MAX_RETRIES,
                        error=str(e),
                    )
                    backoff = DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1)
                    await asyncio.sleep(backoff)

        await self._use_fallback_for_schedule(
            sender_id, session_id, timeout_seconds, metadata
        )

    async def cancel_timer(self, sender_id: Text) -> bool:
        """Cancel a timer by deleting it from Redis.

        This method includes retry logic for transient Redis errors.
        Also checks the fallback store if it's in use.
        """
        # Also cancel from fallback store if in use
        fallback_deleted = False
        if self._using_fallback:
            fallback_deleted = await self._fallback_store.delete_timer(sender_id)

        last_error: Optional[Exception] = None

        for attempt in range(DEFAULT_MAX_RETRIES):
            try:
                deleted = await self._store.delete_timer(sender_id)
                if deleted or fallback_deleted:
                    structlogger.debug(
                        "timer_manager.redis.timer_cancelled",
                        sender_id=sender_id,
                    )
                return deleted or fallback_deleted
            except Exception as e:
                last_error = e
                if attempt < DEFAULT_MAX_RETRIES - 1:
                    structlogger.warning(
                        "timer_manager.redis.cancel_retry",
                        sender_id=sender_id,
                        attempt=attempt + 1,
                        max_retries=DEFAULT_MAX_RETRIES,
                        error=str(e),
                    )
                    backoff = DEFAULT_RETRY_BACKOFF_BASE * (attempt + 1)
                    await asyncio.sleep(backoff)

        # If Redis failed but we deleted from fallback, return True
        if fallback_deleted:
            return True

        structlogger.error(
            "timer_manager.redis.cancel_failed",
            sender_id=sender_id,
            error=str(last_error),
        )
        return False

    async def _claim_and_fire_timer(
        self,
        timer: SessionTimer,
        store: SessionTimerStore,
        log_prefix: Text,
    ) -> bool:
        """Atomically claim a timer and fire the callback if successful.

        Args:
            timer: The timer to process.
            store: The store to delete the timer from.
            log_prefix: Prefix for log event names (e.g., "recovery" or "").

        Returns:
            True if the timer was claimed and callback fired, False otherwise.
        """
        prefix = f"{log_prefix}_" if log_prefix else ""
        try:
            deleted = await store.delete_timer(
                timer.sender_id,
                only_if_scheduled_time=timer.scheduled_time,
            )
            if not deleted:
                structlogger.debug(
                    f"timer_manager.redis.{prefix}timer_not_claimed",
                    sender_id=timer.sender_id,
                )
                return False
            if self._callback:
                await self._callback(
                    timer.sender_id,
                    timer.session_id,
                    timer.scheduled_time,
                )
            return True
        except Exception as e:
            structlogger.error(
                f"timer_manager.redis.{prefix}callback_error",
                sender_id=timer.sender_id,
                error=str(e),
            )
            return False

    async def _recover_expired_timers(self) -> None:
        """Recover and process timers that expired during downtime.

        This is called on startup to handle any timers that should have fired
        while the service was down (e.g., during a restart or deployment).
        """
        try:
            expired_timers = await self._store.get_expired_timers()
            if not expired_timers:
                structlogger.debug("timer_manager.redis.recovery_no_expired_timers")
                return

            structlogger.info(
                "timer_manager.redis.recovery_started",
                expired_count=len(expired_timers),
            )

            processed_count = 0
            for timer in expired_timers:
                if await self._claim_and_fire_timer(timer, self._store, "recovery"):
                    processed_count += 1

            structlogger.info(
                "timer_manager.redis.recovery_completed",
                processed_count=processed_count,
                total_expired=len(expired_timers),
            )
        except Exception as e:
            structlogger.error(
                "timer_manager.redis.recovery_failed",
                error=str(e),
            )

    async def _poll_loop(self) -> None:
        """Continuously poll for and process expired timers.

        This loop is resilient to transient Redis errors - it will log errors
        and continue polling. Consecutive errors are tracked to avoid log spam.
        """
        consecutive_errors = 0
        max_logged_errors = 3  # Only log first N consecutive errors

        while self._running:
            try:
                await self._process_expired_timers()
                if consecutive_errors > 0:
                    structlogger.info(
                        "timer_manager.redis.connection_restored",
                        previous_errors=consecutive_errors,
                    )
                    # Reset fallback flag when Redis recovers
                    if self._using_fallback:
                        self._using_fallback = False
                        structlogger.info(
                            "timer_manager.redis.fallback_deactivated",
                            event_info="Redis connection restored, "
                            "resuming normal operation.",
                        )
                consecutive_errors = 0
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= max_logged_errors:
                    structlogger.error(
                        "timer_manager.redis.poll_error",
                        error=str(e),
                        consecutive_errors=consecutive_errors,
                    )
                elif consecutive_errors == max_logged_errors + 1:
                    structlogger.warning(
                        "timer_manager.redis.poll_errors_suppressed",
                        event_info="Suppressing error logs until connection restored",
                    )
                await asyncio.sleep(self._poll_interval)

    async def _process_expired_timers(self) -> None:
        """Process all currently expired timers from Redis and fallback store."""
        expired_timers = await self._store.get_expired_timers()

        for timer in expired_timers:
            await self._claim_and_fire_timer(timer, self._store, "")

        # Also process fallback store timers if in use
        if self._using_fallback:
            fallback_timers = await self._fallback_store.get_expired_timers()
            for timer in fallback_timers:
                await self._claim_and_fire_timer(
                    timer, self._fallback_store, "fallback"
                )

    async def _use_fallback_for_schedule(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        timeout_seconds: float,
        metadata: Optional[Dict[Text, Any]],
    ) -> None:
        """Fall back to in-memory store when Redis is unavailable."""
        if not self._using_fallback:
            self._using_fallback = True
            structlogger.warning(
                "timer_manager.redis.fallback_activated",
                event_info="Redis unavailable, using in-memory fallback.",
            )

        scheduled_time = time.time() + timeout_seconds
        await self._fallback_store.store_timer(
            sender_id, session_id, scheduled_time, metadata
        )
        structlogger.debug(
            "timer_manager.redis.fallback_timer_scheduled",
            sender_id=sender_id,
            session_id=session_id,
        )
