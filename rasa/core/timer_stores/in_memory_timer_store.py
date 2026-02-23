from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

import structlog

from rasa.core.timer_stores.timer_store import (
    SessionTimer,
    SessionTimerStore,
)

structlogger = structlog.getLogger(__name__)


class InMemorySessionTimerStore(SessionTimerStore):
    """In-memory implementation of SessionTimerStore."""

    def __init__(self) -> None:
        """Initialize the in-memory timer store."""
        self._timers: Dict[Text, SessionTimer] = {}

    def close(self) -> None:
        """Clear all timer data."""
        self._timers.clear()
        structlogger.debug("timer_store.in_memory.closed")

    async def store_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        scheduled_time: float,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Store timer data."""
        timer = SessionTimer.create_timer(
            sender_id, session_id, scheduled_time, metadata
        )
        self._timers[sender_id] = timer
        structlogger.debug(
            "timer_store.in_memory.timer_stored",
            sender_id=sender_id,
            session_id=session_id,
            scheduled_time=scheduled_time,
        )

    async def delete_timer(
        self,
        sender_id: Text,
        only_if_scheduled_time: Optional[float] = None,
    ) -> bool:
        """Delete timer data. Optionally only if scheduled_time matches."""
        timer = self._timers.get(sender_id)
        if self._should_skip_delete(timer, only_if_scheduled_time):
            return False
        del self._timers[sender_id]
        structlogger.debug(
            "timer_store.in_memory.timer_deleted",
            sender_id=sender_id,
        )
        return True

    async def get_timer(self, sender_id: Text) -> Optional[SessionTimer]:
        """Get timer data."""
        return self._timers.get(sender_id)

    async def get_expired_timers(
        self, cutoff_time: Optional[float] = None
    ) -> List[SessionTimer]:
        """Get all expired timers."""
        check_time = self._check_time_for_expired(cutoff_time)
        return self._filter_expired(self._timers.values(), check_time)
