"""
TimerManager implementation for scheduling and executing session timers.

APScheduler was evaluated but not used for the following reasons:

1.  Different usage patterns from existing APScheduler usage in Rasa
    The existing APScheduler usage in the codebase serves different purposes:

    rasa/core/jobs.py: Global singleton AsyncIOScheduler for recurring tasks
    rasa/privacy/privacy_manager.py: Cron-based jobs for anonymization/deletion
    rasa/core/secrets_manager/vault.py: Fixed-interval token refresh

    These are all long-running, stable schedules with few jobs.
    Session timers are per-conversation with high churn
    (create/cancel/reset on every message).

2.  Redis sorted set is the source of truth
    For distributed deployments, we use ZRANGEBYSCORE on a Redis sorted set to
    find expired timers. We're polling a data structure, not scheduling jobs.
    APScheduler's Redis job store would be an additional abstraction layer that
    doesn't match how we query for expirations.

3.  Atomic claim semantics for multi-pod safety
    The delete_timer(only_if_scheduled_time=...) pattern ensures only one pod
    processes each expired timer. When multiple pods poll Redis simultaneously,
    the first to successfully delete "claims" the timer.
    APScheduler doesn't have native claim-or-skip semantics.

4.  Session ID validation before callback execution
    Before firing the callback, we validate the session_id hasn't changed since
    the timer was scheduled.
    This logic would need to wrap any APScheduler job regardless.

5.  Lightweight in-memory implementation
    For single-pod deployments, asyncio.Task with asyncio.sleep is the simplest approach
    for dynamic per-timer scheduling.
    APScheduler would add overhead without benefit for this use case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Text

from rasa.core.timer_stores.timer_store import (
    SessionTimer,
    SessionTimerStore,
    TimerCallback,
    _create_store_from_endpoint_config,
)
from rasa.utils.endpoints import EndpointConfig

DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_BASE = 0.1


class SessionTimerManager(ABC):
    """Abstract base class for timer scheduling and callback execution.

    This class wraps a SessionTimerStore and adds scheduling capabilities.
    It handles:
    - Scheduling timers to fire at the right time
    - Executing callbacks when timers expire
    - Cancelling scheduled timers
    """

    def __init__(self, store: SessionTimerStore) -> None:
        """Initialize the timer manager.

        Args:
            store: The underlying timer store for persistence.
        """
        self._store = store

    @property
    def store(self) -> SessionTimerStore:
        """Get the underlying timer store."""
        return self._store

    @abstractmethod
    async def start(self) -> None:
        """Start the timer manager.

        This should initialize any scheduling infrastructure (e.g., start
        polling loops for Redis-based implementations).
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the timer manager.

        This should clean up any scheduling infrastructure and cancel
        pending timers.
        """
        ...

    @abstractmethod
    async def schedule_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        timeout_seconds: float,
        callback: TimerCallback,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Schedule a timer for a conversation.

        If a timer already exists for this sender_id, it is cancelled first.

        Args:
            sender_id: The conversation ID.
            session_id: The current session ID (for validation when timer fires).
            timeout_seconds: Seconds until the timer fires.
            callback: Async function to call when timer expires.
            metadata: Optional metadata associated with the timer.
        """
        ...

    @abstractmethod
    async def cancel_timer(self, sender_id: Text) -> bool:
        """Cancel a scheduled timer.

        Args:
            sender_id: The conversation ID.

        Returns:
            True if a timer was found and cancelled, False otherwise.
        """
        ...

    async def get_timer(self, sender_id: Text) -> Optional[SessionTimer]:
        """Get timer information for a conversation.

        Args:
            sender_id: The conversation ID.

        Returns:
            SessionTimer if a timer exists, None otherwise.
        """
        return await self._store.get_timer(sender_id)

    @abstractmethod
    def set_callback(self, callback: TimerCallback) -> None:
        """Set the callback to invoke when timers expire.

        For managers that use per-timer callbacks (like InMemorySessionTimerManager),
        this is a no-op. For managers that use a global callback (like
        RedisSessionTimerManager), this sets the callback used for all timer
        expirations.

        Args:
            callback: Async function to call when timer expires.
        """
        ...


def create_timer_manager(
    endpoint_config: Optional[EndpointConfig] = None,
    callback: Optional[TimerCallback] = None,
) -> SessionTimerManager:
    """Create a SessionTimerManager from endpoint configuration.

    Args:
        endpoint_config: Configuration for the timer store.
        callback: Callback to invoke when timers expire (required for Redis).

    Returns:
        Appropriate SessionTimerManager instance.
    """
    from rasa.core.timer_managers.in_memory_timer_manager import (
        InMemorySessionTimerManager,
    )
    from rasa.core.timer_managers.redis_timer_manager import RedisSessionTimerManager
    from rasa.core.timer_stores.redis_timer_store import (
        RedisSessionTimerStore,
        RedisSessionTimerStoreConfig,
    )

    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        return InMemorySessionTimerManager()
    elif endpoint_config.type == "redis":
        config = RedisSessionTimerStoreConfig.model_validate(endpoint_config.to_dict())
        store = RedisSessionTimerStore(config)
        manager = RedisSessionTimerManager(store, config.poll_interval)
        if callback:
            manager.set_callback(callback)
        return manager
    else:
        # For custom stores, wrap with in-memory manager
        custom_store = _create_store_from_endpoint_config(endpoint_config)
        return InMemorySessionTimerManager(custom_store)
