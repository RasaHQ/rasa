"""Tests for planning tools and queue isolation."""

import asyncio

import pytest

from rasa.builder.copilot.agent_sdk.planning_tools import (
    _send_plan_update_to_frontend,
    reset_plan_queue,
    set_plan_queue,
)
from rasa.builder.copilot.models import TodoItem, TodoPlanUpdate


class TestPlanQueueContextIsolation:
    """Tests for ContextVar-based queue isolation.

    These tests verify that the plan queue is properly isolated between
    concurrent requests, preventing plan updates from being sent to the
    wrong user's SSE stream.
    """

    @pytest.mark.asyncio
    async def test_concurrent_requests_use_isolated_queues(self):
        """Test that concurrent requests don't share the plan queue.

        This test verifies the fix for a race condition where:
        1. Request A starts, sets its queue
        2. Request B starts before A completes, sets its queue
        3. Request A's plan updates should still go to A's queue (not B's)

        Using ContextVar ensures each async task has its own queue reference.
        """
        queue_a: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()
        queue_b: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        async def request_a():
            """Simulate Request A."""
            token = set_plan_queue(queue_a)
            try:
                # Request A sends a plan update
                todos_a = [TodoItem(id="A1", content="Task from A", status="pending")]
                _send_plan_update_to_frontend(todos_a)

                # Yield control - Request B starts here and sets its queue
                await asyncio.sleep(0.01)

                # Request A sends another update - should go to queue_a, not queue_b
                todos_a[0].status = "completed"
                _send_plan_update_to_frontend(todos_a)

                return "A done"
            finally:
                reset_plan_queue(token)

        async def request_b():
            """Simulate Request B starting while A is still running."""
            # Small delay to ensure A starts first
            await asyncio.sleep(0.005)

            token = set_plan_queue(queue_b)
            try:
                # Request B sends its own plan update
                todos_b = [TodoItem(id="B1", content="Task from B", status="pending")]
                _send_plan_update_to_frontend(todos_b)

                return "B done"
            finally:
                reset_plan_queue(token)

        # Run both requests concurrently
        results = await asyncio.gather(request_a(), request_b())

        assert results == ["A done", "B done"]

        # Verify queue_a received only Request A's updates
        a_updates = []
        while not queue_a.empty():
            update = queue_a.get_nowait()
            a_updates.append(update)

        assert len(a_updates) == 2
        assert a_updates[0].tasks[0].id == "A1"
        assert a_updates[0].tasks[0].status == "pending"
        assert a_updates[1].tasks[0].id == "A1"
        assert a_updates[1].tasks[0].status == "completed"

        # Verify queue_b received only Request B's updates
        b_updates = []
        while not queue_b.empty():
            update = queue_b.get_nowait()
            b_updates.append(update)

        assert len(b_updates) == 1
        assert b_updates[0].tasks[0].id == "B1"

    @pytest.mark.asyncio
    async def test_set_plan_queue_returns_token(self):
        """Test that set_plan_queue returns a token for cleanup."""
        queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        token = set_plan_queue(queue)

        # Token should be usable for reset
        assert token is not None

        # Clean up
        reset_plan_queue(token)

    @pytest.mark.asyncio
    async def test_send_update_with_no_queue_logs_warning(self):
        """Test that sending updates without a queue logs a warning (doesn't crash)."""
        # No queue set - should not raise, just log warning
        todos = [TodoItem(id="1", content="Test task", status="pending")]

        # This should not raise an exception
        _send_plan_update_to_frontend(todos)

        # No assertion needed - just verify it doesn't crash

    @pytest.mark.asyncio
    async def test_reset_restores_previous_queue_state(self):
        """Test that reset_plan_queue properly restores the previous state."""
        outer_queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()
        inner_queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        # Set outer queue
        outer_token = set_plan_queue(outer_queue)

        # Send to outer
        todos = [TodoItem(id="1", content="Outer task", status="pending")]
        _send_plan_update_to_frontend(todos)

        # Nested: set inner queue
        inner_token = set_plan_queue(inner_queue)

        # Send to inner
        todos = [TodoItem(id="2", content="Inner task", status="pending")]
        _send_plan_update_to_frontend(todos)

        # Reset inner - should restore to outer
        reset_plan_queue(inner_token)

        # Send again - should go to outer
        todos = [TodoItem(id="3", content="Back to outer", status="pending")]
        _send_plan_update_to_frontend(todos)

        # Clean up outer
        reset_plan_queue(outer_token)

        # Verify outer queue got 2 updates (before inner and after reset)
        outer_updates = []
        while not outer_queue.empty():
            outer_updates.append(outer_queue.get_nowait())

        assert len(outer_updates) == 2
        assert outer_updates[0].tasks[0].id == "1"
        assert outer_updates[1].tasks[0].id == "3"

        # Verify inner queue got 1 update
        inner_updates = []
        while not inner_queue.empty():
            inner_updates.append(inner_queue.get_nowait())

        assert len(inner_updates) == 1
        assert inner_updates[0].tasks[0].id == "2"
