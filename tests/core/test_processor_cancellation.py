"""Tests for the MessageProcessor cancellation token registry."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.agents.core.cancellation import CancellationToken
from rasa.shared.core.events import (
    AgentCancelled,
    AgentStarted,
    ConversationInactive,
    FlowCancelled,
    SessionStarted,
    UserUttered,
)


def _create_processor_with_registry():
    """Create a minimal mock of MessageProcessor with the registry methods."""
    processor = MagicMock()
    processor._active_cancellation_tokens = {}

    from rasa.core.processor import MessageProcessor

    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.unregister_cancellation_token = (
        MessageProcessor.unregister_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )
    return processor


def test_cancel_background_tasks_signals_token():
    processor = _create_processor_with_registry()
    token = CancellationToken()
    processor.register_cancellation_token("conv-1", token)

    result = processor.cancel_background_tasks("conv-1")

    assert result is True
    assert token.is_cancelled is True


def test_cancel_background_tasks_returns_false_when_no_token():
    processor = _create_processor_with_registry()

    result = processor.cancel_background_tasks("unknown-sender")

    assert result is False


def test_unregister_cleans_up_token():
    processor = _create_processor_with_registry()
    token = CancellationToken()
    processor.register_cancellation_token("conv-1", token)
    processor.unregister_cancellation_token("conv-1")

    result = processor.cancel_background_tasks("conv-1")
    assert result is False
    assert token.is_cancelled is False


def test_register_overwrites_previous_token():
    processor = _create_processor_with_registry()
    token1 = CancellationToken()
    token2 = CancellationToken()
    processor.register_cancellation_token("conv-1", token1)
    processor.register_cancellation_token("conv-1", token2)

    result = processor.cancel_background_tasks("conv-1")

    assert result is True
    assert token2.is_cancelled is True
    assert token1.is_cancelled is False


def test_unregister_nonexistent_is_safe():
    processor = _create_processor_with_registry()
    processor.unregister_cancellation_token("nonexistent")


# =============================================================================
# End-to-end: Agent.handle_message lifecycle
# =============================================================================

FLOW_ID = "test_flow"
AGENT_NAME = "test_agent"
STEP_ID = "call_agent_step"


def _create_agent_with_tracker_store():
    """Create an Agent with real token lifecycle and an in-memory tracker store.

    The tracker store persists events across calls, so we can assert on
    the full event sequence after cancellation + resume.
    """
    from rasa.core.agent import Agent
    from rasa.core.lock_store import InMemoryLockStore
    from rasa.core.processor import MessageProcessor
    from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
    from rasa.shared.core.domain import Domain

    domain = Domain.empty()
    tracker_store = InMemoryTrackerStore(domain)

    processor = MagicMock(spec=MessageProcessor)
    processor._active_cancellation_tokens = {}
    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.unregister_cancellation_token = (
        MessageProcessor.unregister_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )
    processor.tracker_store = tracker_store
    processor.domain = domain
    processor.lock_store = InMemoryLockStore()

    agent = Agent.__new__(Agent)
    agent.processor = processor
    agent.tracker_store = tracker_store
    agent.lock_store = processor.lock_store
    return agent, processor, tracker_store


@pytest.mark.asyncio
async def test_e2e_events_on_tracker_after_cancel_and_resume():
    """Verify domain events on the tracker across cancel → inactive → resume.

    Simulates the full lifecycle through Agent.handle_message while recording
    events to a real InMemoryTrackerStore:

    1. Message 1 → AgentStarted → (A2A blocks) → session timeout fires
       cancel_background_tasks → AgentCancelled + FlowCancelled written to
       tracker → ConversationInactive written to tracker.
    2. Message 2 → new session starts → AgentStarted → A2A completes → done.

    Asserts the presence and ordering of all key events.
    """
    from rasa.core.channels.channel import UserMessage

    agent, processor, tracker_store = _create_agent_with_tracker_store()
    sender_id = "user-e2e-events"

    # Pre-populate tracker with an initial session so handle_session_timeout
    # finds a valid conversation.
    tracker = await tracker_store.get_or_create_tracker(sender_id)
    tracker.update(SessionStarted())
    tracker.update(UserUttered("setup"))
    await tracker_store.save(tracker)

    async def _msg1_handle_message(message):
        """Simulate: AgentStarted → block on token → on cancel write events."""
        token = processor._active_cancellation_tokens.get(message.sender_id)
        t = await tracker_store.get_or_create_tracker(sender_id)

        t.update(AgentStarted(agent_id=AGENT_NAME, flow_id=FLOW_ID))
        await tracker_store.save(t)

        cancelled = await token.wait(timeout=30)
        if cancelled:
            t = await tracker_store.get_or_create_tracker(sender_id)
            t.update(
                AgentCancelled(
                    agent_id=AGENT_NAME,
                    flow_id=FLOW_ID,
                    reason="Streaming cancelled",
                )
            )
            t.update(FlowCancelled(FLOW_ID, STEP_ID))
            await tracker_store.save(t)
            return []

        return [{"response": "done"}]

    async def _session_timeout():
        """Simulate session timeout: cancel → lock → ConversationInactive."""
        await asyncio.sleep(0.1)
        agent.cancel_background_tasks(sender_id)
        # After the lock is released by handle_message, apply ConversationInactive
        # (in production this happens inside handle_session_timeout which also
        # acquires the lock — here we wait for it to be free).
        async with agent.lock_store.lock(sender_id):
            t = await tracker_store.get_or_create_tracker(sender_id)
            t.update(ConversationInactive())
            await tracker_store.save(t)

    processor.handle_message = AsyncMock(side_effect=_msg1_handle_message)

    timeout_task = asyncio.create_task(_session_timeout())
    await agent.handle_message(UserMessage("Hello", sender_id=sender_id))
    await timeout_task

    # --- Assert events after cancellation + inactivity ---
    tracker = await tracker_store.get_or_create_tracker(sender_id)
    events = list(tracker.events)

    def _find(event_type, start=0):
        """Return the index of the first event of the given type from start."""
        for i in range(start, len(events)):
            if isinstance(events[i], event_type):
                return i
        return None

    agent_started_idx = _find(AgentStarted)
    agent_cancelled_idx = _find(AgentCancelled)
    flow_cancelled_idx = _find(FlowCancelled)
    inactive_idx = _find(ConversationInactive)

    assert agent_started_idx is not None
    assert agent_cancelled_idx is not None
    assert flow_cancelled_idx is not None
    assert inactive_idx is not None

    assert agent_started_idx < agent_cancelled_idx
    assert agent_cancelled_idx < flow_cancelled_idx
    assert flow_cancelled_idx < inactive_idx

    assert tracker.inactive is True

    cancelled_event = events[agent_cancelled_idx]
    assert cancelled_event.agent_id == AGENT_NAME
    assert cancelled_event.flow_id == FLOW_ID
    assert cancelled_event.reason == "Streaming cancelled"

    # --- Message 2: resume the conversation ---
    async def _msg2_handle_message(message):
        """Simulate resumed processing: new session + agent completes."""
        token = processor._active_cancellation_tokens.get(message.sender_id)
        assert token.is_cancelled is False

        t = await tracker_store.get_or_create_tracker(sender_id)
        t.update(SessionStarted())
        t.update(UserUttered("Resume"))
        t.update(AgentStarted(agent_id=AGENT_NAME, flow_id=FLOW_ID))
        await tracker_store.save(t)
        return [{"response": "resumed"}]

    processor.handle_message = AsyncMock(side_effect=_msg2_handle_message)
    result2 = await agent.handle_message(UserMessage("Resume", sender_id=sender_id))
    assert result2 == [{"response": "resumed"}]

    # --- Assert events after resume ---
    tracker = await tracker_store.get_or_create_tracker(sender_id)
    events = list(tracker.events)

    # ConversationInactive should still be present exactly once
    assert sum(1 for e in events if isinstance(e, ConversationInactive)) == 1

    # After the inactive event, there should be a new SessionStarted + AgentStarted
    new_session_idx = _find(SessionStarted, inactive_idx + 1)
    new_agent_idx = _find(AgentStarted, inactive_idx + 1)

    assert new_session_idx is not None
    assert new_agent_idx is not None
    assert new_session_idx < new_agent_idx

    # Tracker is no longer inactive after the new session
    assert tracker.inactive is False


@pytest.mark.asyncio
async def test_e2e_session_timeout_stops_a2a_polling_and_marks_inactive():
    """Session expiration shorter than A2A max polling time stops polling.

    Wires together a real A2A polling loop (with a mock A2A client that never
    completes) and the real ``handle_session_timeout`` method.  Verifies:

    1. ``cancel_background_tasks`` interrupts the polling before max_wait.
    2. ``ConversationInactive`` is persisted on the tracker.
    3. The A2A output is ``CANCELLED`` (not a timeout / fatal error).
    """
    import time as time_mod
    from unittest.mock import patch

    from a2a.types import Task, TaskState, TaskStatus

    from rasa.agents.core.cancellation import CancellationToken
    from rasa.agents.core.types import AgentStatus
    from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
    from rasa.agents.schemas import AgentInput
    from rasa.core.available_agents import (
        AgentConfig,
        AgentConfiguration,
        AgentInfo,
        ProtocolConfig,
    )
    from rasa.core.lock_store import InMemoryLockStore
    from rasa.core.processor import MessageProcessor
    from rasa.core.tracker_stores.tracker_store import InMemoryTrackerStore
    from rasa.shared.core.domain import Domain

    # -- Set up processor with real methods for the timeout path --
    domain = Domain.empty()
    tracker_store = InMemoryTrackerStore(domain)
    lock_store = InMemoryLockStore()

    processor = MagicMock(spec=MessageProcessor)
    processor._active_cancellation_tokens = {}
    processor.register_cancellation_token = (
        MessageProcessor.register_cancellation_token.__get__(processor)
    )
    processor.unregister_cancellation_token = (
        MessageProcessor.unregister_cancellation_token.__get__(processor)
    )
    processor.cancel_background_tasks = (
        MessageProcessor.cancel_background_tasks.__get__(processor)
    )
    processor.handle_session_timeout = MessageProcessor.handle_session_timeout.__get__(
        processor
    )
    processor.get_tracker = MessageProcessor.get_tracker.__get__(processor)
    processor.save_tracker = MessageProcessor.save_tracker.__get__(processor)
    processor.tracker_store = tracker_store
    processor.lock_store = lock_store
    processor.domain = domain
    processor.model_metadata = MagicMock(model_id="test", assistant_id="test")
    processor.model_filename = "test_model"

    sender_id = "user-poll-timeout"

    # Pre-populate tracker with a valid session
    tracker = await tracker_store.get_or_create_tracker(sender_id)
    tracker.update(SessionStarted())
    tracker.update(UserUttered("hello"))
    await tracker_store.save(tracker)
    session_id = tracker.current_session_id

    # -- Set up A2A agent with a client that never reaches terminal state --
    non_terminal_task = Task(
        context_id="ctx",
        id="ctx-001",
        status=TaskStatus(state=TaskState.working),
    )

    async def _forever_working_stream():
        yield (non_terminal_task, None)

    mock_client = MagicMock()
    mock_client.send_message.side_effect = lambda *a, **kw: _forever_working_stream()
    mock_client.get_task = AsyncMock(return_value=non_terminal_task)

    with patch(
        "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._init_client"
    ) as mock_init_client:
        mock_init_client.return_value = mock_client
        a2a_agent = A2AAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name=AGENT_NAME,
                    description="Test",
                    protocol=ProtocolConfig.A2A,
                ),
                configuration=AgentConfiguration(agent_card="some/path"),
            )
        )
        with patch(
            "rasa.agents.protocol.a2a.a2a_agent.A2AAgent._load_agent_card_from_file"
        ) as mock_load_card:
            mock_card = MagicMock()
            mock_card.url = "http://example.com"
            mock_load_card.return_value = mock_card
            await a2a_agent.connect()

    # Force a large max_wait so polling would normally run for a long time
    import rasa.agents.protocol.a2a.a2a_agent as a2a_mod

    original_max_wait = getattr(a2a_mod, "A2A_TASK_POLLING_MAX_WAIT", 60)

    token = CancellationToken()
    processor.register_cancellation_token(sender_id, token)

    async def _fire_session_timeout():
        """After a short delay, fire handle_session_timeout — the same path
        the real timer manager would invoke."""
        await asyncio.sleep(0.15)
        await processor.handle_session_timeout(sender_id, session_id)

    timeout_task = asyncio.create_task(_fire_session_timeout())

    try:
        a2a_mod.A2A_TASK_POLLING_MAX_WAIT = 120

        start = time_mod.monotonic()
        output = await a2a_agent.run(
            AgentInput(
                id="ctx",
                metadata={},
                user_message="Test",
                slots=[],
                conversation_history="",
                events=[],
            ),
            cancellation_token=token,
        )
        elapsed = time_mod.monotonic() - start
    finally:
        a2a_mod.A2A_TASK_POLLING_MAX_WAIT = original_max_wait
        processor.unregister_cancellation_token(sender_id)

    await timeout_task

    # -- A2A polling was interrupted, not timed out --
    assert output.status == AgentStatus.CANCELLED
    assert (output.metadata or {}).get("cancellation_reason") == "Polling cancelled"
    assert elapsed < 5.0, f"Polling should have exited promptly, took {elapsed:.2f}s"

    # -- ConversationInactive was applied by handle_session_timeout --
    tracker = await tracker_store.get_or_create_tracker(sender_id)
    events = list(tracker.events)

    inactive_events = [e for e in events if isinstance(e, ConversationInactive)]
    assert len(inactive_events) == 1
    assert tracker.inactive is True
    assert timeout_task.done()
