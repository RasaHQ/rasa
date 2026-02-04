"""Tests for planning tools and queue isolation."""

import asyncio

import pytest

from rasa.builder.copilot.agent_sdk.planning_tools import (
    _send_plan_update_to_frontend,
    reset_plan_queue,
    set_plan_queue,
)
from rasa.builder.copilot.models import (
    TaskStatusUpdate,
    TodoItem,
    TodoPlanUpdate,
)


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


class TestTaskStatusUpdates:
    """Tests for task status updates including the failed status."""

    @pytest.mark.asyncio
    async def test_todo_item_supports_failed_status(self):
        """Test that TodoItem can be created with failed status."""
        task = TodoItem(id="1", content="Test task", status="failed")

        assert task.status == "failed"
        assert task.id == "1"
        assert task.content == "Test task"

    @pytest.mark.asyncio
    async def test_task_status_update_supports_failed_status(self):
        """Test that TaskStatusUpdate can be created with failed status."""
        update = TaskStatusUpdate(task_id="1", status="failed")

        assert update.status == "failed"
        assert update.task_id == "1"

    @pytest.mark.asyncio
    async def test_task_status_transition_to_failed(self):
        """Test that a task can transition from in_progress to failed.

        This test verifies the model correctly handles the failed status
        when simulating a task that encounters an error.
        """
        queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        token = set_plan_queue(queue)
        try:
            # Simulate a plan where task 1 was in_progress and then failed
            todos = [
                TodoItem(id="1", content="Validate project", status="in_progress"),
                TodoItem(id="2", content="Fix errors", status="pending"),
                TodoItem(id="3", content="Train model", status="pending"),
            ]

            # Send initial state
            _send_plan_update_to_frontend(todos)

            # Simulate task failure
            todos[0].status = "failed"
            _send_plan_update_to_frontend(todos)

            # Verify both updates were received
            updates = []
            while not queue.empty():
                updates.append(queue.get_nowait())

            assert len(updates) == 2

            # First update: in_progress
            assert updates[0].tasks[0].status == "in_progress"

            # Second update: failed
            assert updates[1].tasks[0].status == "failed"
            assert updates[1].tasks[1].status == "pending"
            assert updates[1].tasks[2].status == "pending"

        finally:
            reset_plan_queue(token)

    @pytest.mark.asyncio
    async def test_failed_status_in_plan_update_event(self):
        """Test that failed status is correctly sent to frontend via queue."""
        queue: asyncio.Queue[TodoPlanUpdate] = asyncio.Queue()

        token = set_plan_queue(queue)
        try:
            # Create a task with failed status
            todos = [
                TodoItem(id="1", content="Validate project", status="failed"),
                TodoItem(id="2", content="Fix errors", status="pending"),
            ]

            _send_plan_update_to_frontend(todos)

            # Verify the update was sent
            assert not queue.empty()
            update = queue.get_nowait()

            assert len(update.tasks) == 2
            assert update.tasks[0].status == "failed"
            assert update.tasks[1].status == "pending"

        finally:
            reset_plan_queue(token)

    @pytest.mark.asyncio
    async def test_all_task_statuses_are_valid(self):
        """Test that all defined task statuses can be used."""
        valid_statuses = ["pending", "in_progress", "completed", "cancelled", "failed"]

        for status in valid_statuses:
            # Should not raise validation error
            task = TodoItem(id="1", content="Test", status=status)
            assert task.status == status

            update = TaskStatusUpdate(task_id="1", status=status)
            assert update.status == status
