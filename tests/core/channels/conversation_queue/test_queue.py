"""Unit tests for ConversationQueue."""

import asyncio
from typing import Any

import pytest

from rasa.core.channels.conversation_queue.events import (
    InputEvent,
)
from rasa.core.channels.conversation_queue.queue import (
    ConversationQueue,
    InMemoryConversationQueue,
)


# Test event classes
class DummyInputEvent(InputEvent):
    """Dummy event for testing."""

    def __init__(self, data: str) -> None:
        self.data = data


class AnotherInputEvent(InputEvent):
    """Another dummy event for testing."""

    def __init__(self, value: int) -> None:
        self.value = value


# Fixtures for InMemoryConversationQueue tests
@pytest.fixture
def queue() -> InMemoryConversationQueue[str]:
    """Create a queue for testing."""
    return InMemoryConversationQueue[str](
        conversation_id="test-conversation",
        input_channel="test_channel",
        maxsize=10,
    )


# Tests for InMemoryConversationQueue
@pytest.mark.asyncio
async def test_queue_initialization() -> None:
    """Test that queue is initialized with correct properties."""
    queue = InMemoryConversationQueue[str](
        conversation_id="test-123",
        input_channel="test_channel",
        maxsize=50,
    )
    assert queue.conversation_id == "test-123"
    assert queue.input_channel == "test_channel"


@pytest.mark.asyncio
async def test_put_and_get(queue: InMemoryConversationQueue[str]) -> None:
    """Test basic put and get operations."""
    await queue.put("event1")
    await queue.put("event2")

    event1 = await queue.get()
    event2 = await queue.get()

    assert event1 == "event1"
    assert event2 == "event2"


@pytest.mark.asyncio
async def test_fifo_order(queue: InMemoryConversationQueue[str]) -> None:
    """Test that events are retrieved in FIFO order."""
    events = ["first", "second", "third", "fourth"]

    for event in events:
        await queue.put(event)

    retrieved = []
    for _ in range(len(events)):
        retrieved.append(await queue.get())

    assert retrieved == events


@pytest.mark.asyncio
async def test_drain_empty_queue(queue: InMemoryConversationQueue[str]) -> None:
    """Test that draining an empty queue returns an empty list."""
    events = await queue.drain()
    assert events == []


@pytest.mark.asyncio
async def test_drain_returns_all_events(queue: InMemoryConversationQueue[str]) -> None:
    """Test that drain returns all events without blocking."""
    await queue.put("event1")
    await queue.put("event2")
    await queue.put("event3")

    events = await queue.drain()

    assert len(events) == 3
    assert events == ["event1", "event2", "event3"]


@pytest.mark.asyncio
async def test_drain_empties_queue(queue: InMemoryConversationQueue[str]) -> None:
    """Test that drain empties the queue."""
    await queue.put("event1")
    await queue.put("event2")

    await queue.drain()

    # Queue should be empty after drain
    events = await queue.drain()
    assert events == []


@pytest.mark.asyncio
async def test_drain_preserves_order(queue: InMemoryConversationQueue[str]) -> None:
    """Test that drain preserves FIFO order."""
    events = ["first", "second", "third", "fourth", "fifth"]

    for event in events:
        await queue.put(event)

    drained = await queue.drain()
    assert drained == events


@pytest.mark.asyncio
async def test_get_blocks_when_empty(queue: InMemoryConversationQueue[str]) -> None:
    """Test that get() blocks when queue is empty."""
    # Create a task that will get from empty queue
    get_task = asyncio.create_task(queue.get())

    # Give it a tiny moment to start waiting
    await asyncio.sleep(0.01)

    # Task should not be done yet
    assert not get_task.done()

    # Put an event
    await queue.put("event")

    # Now the task should complete
    result = await get_task
    assert result == "event"


@pytest.mark.asyncio
async def test_get_batch_single_item(queue: InMemoryConversationQueue[str]) -> None:
    """get_batch returns a one-element list when only one event is queued."""
    await queue.put("only")
    batch = await queue.get_batch()
    assert batch == ["only"]


@pytest.mark.asyncio
async def test_get_batch_drains_rest(queue: InMemoryConversationQueue[str]) -> None:
    """get_batch returns FIFO order: first waited item then drained remainder."""
    await queue.put("a")
    await queue.put("b")
    await queue.put("c")
    batch = await queue.get_batch()
    assert batch == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_get_batch_blocks_until_first_item(
    queue: InMemoryConversationQueue[str],
) -> None:
    """get_batch blocks on an empty queue until at least one put completes."""
    batch_task = asyncio.create_task(queue.get_batch())
    await asyncio.sleep(0.01)
    assert not batch_task.done()

    await queue.put("first")
    await queue.put("second")

    batch = await batch_task
    assert batch == ["first", "second"]


@pytest.mark.asyncio
async def test_maxsize_enforcement() -> None:
    """Test that maxsize limits queue capacity."""
    queue = InMemoryConversationQueue[str](
        conversation_id="test",
        input_channel="test_channel",
        maxsize=2,
    )

    # Fill the queue
    await queue.put("event1")
    await queue.put("event2")

    # Try to put another - this should block
    put_task = asyncio.create_task(queue.put("event3"))

    # Give it a moment to try
    await asyncio.sleep(0.01)

    # Task should not be done (blocking on full queue)
    assert not put_task.done()

    # Get one item to make space
    await queue.get()

    # Now the put should complete
    await put_task
    assert put_task.done()


@pytest.mark.asyncio
async def test_queue_with_complex_types() -> None:
    """Test that queue works with complex event types."""
    queue = InMemoryConversationQueue[dict[str, Any]](
        conversation_id="test",
        input_channel="test_channel",
        maxsize=10,
    )

    event1 = {"type": "transcript", "text": "hello"}
    event2 = {"type": "interruption", "timestamp": 12345}

    await queue.put(event1)
    await queue.put(event2)

    retrieved1 = await queue.get()
    retrieved2 = await queue.get()

    assert retrieved1 == event1
    assert retrieved2 == event2


@pytest.mark.asyncio
async def test_protocol_compliance() -> None:
    """Test that InMemoryConversationQueue implements the protocol."""
    queue = InMemoryConversationQueue[str](
        conversation_id="test",
        input_channel="test_channel",
        maxsize=10,
    )

    # Check that it's recognized as a ConversationQueue
    assert isinstance(queue, ConversationQueue)


@pytest.mark.asyncio
async def test_queue_with_input_events() -> None:
    """Test that queue works with InputEvent subclasses."""
    queue = InMemoryConversationQueue[InputEvent](
        conversation_id="test-conv",
        input_channel="test_channel",
        maxsize=10,
    )

    event1 = DummyInputEvent("transcript")
    event2 = AnotherInputEvent(123)

    await queue.put(event1)
    await queue.put(event2)

    retrieved1 = await queue.get()
    retrieved2 = await queue.get()

    assert isinstance(retrieved1, DummyInputEvent)
    assert isinstance(retrieved2, AnotherInputEvent)
    assert retrieved1.data == "transcript"
    assert retrieved2.value == 123
