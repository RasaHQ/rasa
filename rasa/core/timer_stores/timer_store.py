"""Base timer store and factory.

This module defines the abstract SessionTimerStore, shared types (SessionTimer,
TimerCallback), and the factory for creating stores from endpoint config.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Text,
    Union,
)

import structlog
from pydantic import BaseModel, Field

import rasa.shared.utils.common
from rasa.shared.exceptions import ConnectionException
from rasa.utils.endpoints import EndpointConfig

structlogger = structlog.getLogger(__name__)

# Type alias for timer callback functions
# (sender_id, session_id, optional scheduled_time)
TimerCallback = Callable[
    [Text, Optional[Text], Optional[float]], Coroutine[Any, Any, None]
]


class SessionTimer(BaseModel):
    """Represents a scheduled session timer.

    Attributes:
        sender_id: The conversation ID this timer is associated with.
        session_id: The session ID when the timer was scheduled (for validation).
        scheduled_time: Unix timestamp when the timer should fire.
        metadata: Optional metadata associated with the timer.
    """

    sender_id: Text
    session_id: Optional[Text] = None
    scheduled_time: float
    metadata: Dict[Text, Any] = Field(default_factory=dict)

    def as_dict(self) -> Dict[Text, Any]:
        """Serialize timer to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[Text, Any]) -> SessionTimer:
        """Create SessionTimer from dictionary."""
        return cls.model_validate(data)

    @classmethod
    def create_timer(
        cls,
        sender_id: Text,
        session_id: Optional[Text],
        scheduled_time: float,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> SessionTimer:
        """Build a SessionTimer from store_timer arguments (shared by all stores)."""
        return cls(
            sender_id=sender_id,
            session_id=session_id,
            scheduled_time=scheduled_time,
            metadata=metadata or {},
        )


class SessionTimerStore(ABC):
    """Base class for timer stores."""

    @staticmethod
    def create(
        obj: Union[SessionTimerStore, EndpointConfig, None],
    ) -> SessionTimerStore:
        """Factory to create a timer store from an existing store or endpoint config."""
        if isinstance(obj, SessionTimerStore):
            return obj

        try:
            return _create_store_from_endpoint_config(obj)
        except ConnectionError as error:
            raise ConnectionException("Cannot connect to timer store.") from error

    @abstractmethod
    def close(self) -> None:
        """Close the timer store connection.

        Subclasses must implement this method to handle any necessary cleanup.
        """
        ...

    @abstractmethod
    async def store_timer(
        self,
        sender_id: Text,
        session_id: Optional[Text],
        scheduled_time: float,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> None:
        """Store timer data for a conversation.

        Args:
            sender_id: The conversation ID.
            session_id: The current session ID (for validation when timer fires).
            scheduled_time: Unix timestamp when the timer should fire.
            metadata: Optional metadata associated with the timer.
        """
        ...

    @abstractmethod
    async def delete_timer(
        self,
        sender_id: Text,
        only_if_scheduled_time: Optional[float] = None,
    ) -> bool:
        """Delete timer data for a conversation.

        Args:
            sender_id: The conversation ID.
            only_if_scheduled_time: If set, delete only when the stored timer's
                scheduled_time equals this value (atomic claim in multi-pod).
                If None, delete unconditionally.

        Returns:
            True if a timer was found and deleted, False otherwise.
        """
        ...

    @abstractmethod
    async def get_timer(self, sender_id: Text) -> Optional[SessionTimer]:
        """Get timer information for a conversation.

        Args:
            sender_id: The conversation ID.

        Returns:
            SessionTimer if a timer exists for this conversation, None otherwise.
        """
        ...

    @abstractmethod
    async def get_expired_timers(
        self, cutoff_time: Optional[float] = None
    ) -> List[SessionTimer]:
        """Return timers that have expired (scheduled_time <= cutoff_time).

        Used by the timer manager to discover which timers are due so it can
        invoke the session-timeout callback and then delete them (clearing
        happens via delete_timer in the caller, not in this method).

        Args:
            cutoff_time: Unix timestamp threshold. Timers with
                scheduled_time <= cutoff_time are returned. Defaults to
                current time if None.

        Returns:
            List of expired SessionTimer objects.
        """
        ...

    @staticmethod
    def _check_time_for_expired(cutoff_time: Optional[float]) -> float:
        """Normalize cutoff_time for get_expired_timers.

        Used by store implementations to handle the optional cutoff_time
        parameter, defaulting to the current time when None is passed.
        """
        return cutoff_time if cutoff_time is not None else time.time()

    @staticmethod
    def _filter_expired(
        timers: Iterable[SessionTimer], check_time: float
    ) -> List[SessionTimer]:
        """Filter timers to only those that have expired.

        Used by store implementations to select timers whose scheduled_time
        is at or before the check_time threshold.
        """
        return [t for t in timers if t.scheduled_time <= check_time]

    @staticmethod
    def _should_skip_delete(
        timer: Optional[SessionTimer],
        only_if_scheduled_time: Optional[float],
    ) -> bool:
        """Check if a delete operation should be skipped.

        Used by store implementations to enforce atomic delete semantics.
        Returns True if the timer doesn't exist or if only_if_scheduled_time
        is specified but doesn't match the timer's scheduled_time.
        """
        if timer is None:
            return True
        if only_if_scheduled_time is not None and (
            timer.scheduled_time != only_if_scheduled_time
        ):
            return True
        return False


def _create_store_from_endpoint_config(
    endpoint_config: Optional[EndpointConfig] = None,
) -> SessionTimerStore:
    """Given an endpoint configuration, create a proper `SessionTimerStore` object."""
    # Import implementations lazily to avoid circular imports
    from rasa.core.timer_stores.in_memory_timer_store import InMemorySessionTimerStore
    from rasa.core.timer_stores.redis_timer_store import (
        RedisSessionTimerStore,
        RedisSessionTimerStoreConfig,
    )

    if (
        endpoint_config is None
        or endpoint_config.type is None
        or endpoint_config.type == "in_memory"
    ):
        timer_store: SessionTimerStore = InMemorySessionTimerStore()
    elif endpoint_config.type == "redis":
        config = RedisSessionTimerStoreConfig.model_validate(endpoint_config.to_dict())
        timer_store = RedisSessionTimerStore(config)
    else:
        timer_store = _load_from_module_name_in_endpoint_config(endpoint_config)

    structlogger.debug(
        "timer_store._create_from_endpoint_config.timer_store_connected",
        event_info=f"Connected to timer store '{timer_store.__class__.__name__}'.",
    )

    return timer_store


def _load_from_module_name_in_endpoint_config(
    endpoint_config: EndpointConfig,
) -> SessionTimerStore:
    """Retrieve a `SessionTimerStore` based on its class name."""
    try:
        timer_store_class = rasa.shared.utils.common.class_from_module_path(
            endpoint_config.type
        )
        return timer_store_class(endpoint_config=endpoint_config)
    except (AttributeError, ImportError) as e:
        raise Exception(
            f"Could not find a class based on the module path "
            f"'{endpoint_config.type}'. Failed to create a `SessionTimerStore` "
            f"instance. Error: {e}"
        ) from e
