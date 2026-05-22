"""Conversation queue for event-based channel communication."""

from __future__ import annotations

import asyncio
from typing import Generic, Protocol, TypeVar, runtime_checkable

import structlog

structlogger = structlog.get_logger()

T = TypeVar("T")


@runtime_checkable
class ConversationQueue(Protocol[T]):
    """Typed async event queue for conversation events.

    Default implementation wraps asyncio.Queue. This protocol allows for
    distributed backends (Redis Streams, Kafka) to be swapped in without
    changing call sites.

    The protocol is generic over event type T to allow different channels to use
    different event hierarchies (e.g., voice events vs text events), though in
    practice most will use InputEvent or its subclasses.
    """

    @property
    def conversation_id(self) -> str:
        """The conversation ID this queue belongs to."""
        ...

    @property
    def input_channel(self) -> str:
        """The input channel name that produced events for this queue."""
        ...

    async def put(self, event: T) -> None:
        """Add an event to the queue.

        Args:
            event: The event to enqueue.
        """
        ...

    async def get(self) -> T:
        """Get the next event from the queue.

        Blocks until an event is available.

        Returns:
            The next event in the queue.
        """
        ...

    async def drain(self) -> list[T]:
        """Drain all available events from the queue.

        Returns all events currently in the queue without blocking.
        If the queue is empty, returns an empty list.

        This is used to prevent the bot from responding to stale input
        by processing all pending events in a single batch.

        Returns:
            List of all events currently in the queue.
        """
        ...

    async def get_batch(self) -> list[T]:
        """Block until the first event, then drain the rest without blocking.

        Waits on `get()` for the first item (so the caller does not busy-spin on an
        empty queue), then `drain()` for any further items already queued. The returned
        list is never empty when this coroutine completes.

        Returns:
            Non-empty list `[first, *rest]` for this batch.
        """
        ...


class InMemoryConversationQueue(Generic[T]):
    """In-memory implementation of ConversationQueue using asyncio.Queue.

    This is the default implementation for voice conversations. For most use cases,
    an in-memory queue is sufficient and distributed backends are not needed.

    Args:
        conversation_id: The conversation ID this queue belongs to.
        input_channel: The input channel name that produced events for this queue.
        maxsize: Maximum number of events the queue can hold. If maxsize is 0,
            the queue size is unlimited.
    """

    def __init__(
        self, conversation_id: str, input_channel: str, maxsize: int = 50
    ) -> None:
        """Initialize the queue.

        Args:
            conversation_id: The conversation ID this queue belongs to.
            input_channel: The input channel name that produced events for this queue.
            maxsize: Maximum number of events the queue can hold. Defaults to 50.
        """
        self._conversation_id = conversation_id
        self._input_channel = input_channel
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)

        structlogger.debug(
            "conversation_queue.created",
            conversation_id=conversation_id,
            input_channel=input_channel,
            maxsize=maxsize,
        )

    @property
    def conversation_id(self) -> str:
        """The conversation ID this queue belongs to."""
        return self._conversation_id

    @property
    def input_channel(self) -> str:
        """The input channel name that produced events for this queue."""
        return self._input_channel

    async def put(self, event: T) -> None:
        """Add an event to the queue.

        Blocks if the queue is full (when maxsize > 0).

        Args:
            event: The event to enqueue.
        """
        await self._queue.put(event)
        structlogger.debug(
            "conversation_queue.event_enqueued",
            conversation_id=self._conversation_id,
            event_type=type(event).__name__,
            queue_size=self._queue.qsize(),
        )

    async def get(self) -> T:
        """Get the next event from the queue.

        Blocks until an event is available.

        Returns:
            The next event in the queue.
        """
        event = await self._queue.get()
        structlogger.debug(
            "conversation_queue.event_dequeued",
            conversation_id=self._conversation_id,
            event_type=type(event).__name__,
            queue_size=self._queue.qsize(),
        )
        return event

    async def drain(self) -> list[T]:
        """Drain all available events from the queue.

        Returns all events currently in the queue without blocking.
        If the queue is empty, returns an empty list.

        This prevents the bot from responding to stale input by processing
        all pending events in a single batch.

        Returns:
            List of all events currently in the queue.
        """
        events: list[T] = []

        # Get all events from the queue without blocking.
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                events.append(event)
            except asyncio.QueueEmpty:
                # Race condition: queue became empty between check and get
                break

        if events:
            structlogger.debug(
                "conversation_queue.drained",
                conversation_id=self._conversation_id,
                event_count=len(events),
                event_types=[type(e).__name__ for e in events],
            )

        return events

    async def get_batch(self) -> list[T]:
        """Block until the first event, then drain the rest without blocking.

        Waits on `get()` for the first item (so the caller does not busy-spin on an
        empty queue), then `drain()` for any further items already queued. The returned
        list is never empty when this coroutine completes.

        Returns:
            Non-empty list `[first, *rest]` for this batch.
        """
        first = await self.get()
        rest = await self.drain()
        batch = [first, *rest]
        structlogger.debug(
            "conversation_queue.get_batch",
            conversation_id=self._conversation_id,
            event_count=len(batch),
            event_types=[type(e).__name__ for e in batch],
        )
        return batch
