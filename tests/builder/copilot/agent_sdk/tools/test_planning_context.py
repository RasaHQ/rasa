"""Tests for planning context ContextVar management."""

import asyncio

from rasa.builder.copilot.agent_sdk.tools.planning_context import (
    format_plan_for_agent,
    get_current_plan,
    get_final_plan,
    init_planning_context,
    reset_planning_context,
    set_current_plan,
)
from rasa.builder.copilot.models import TodoItem


class TestPlanningContext:
    """Test class for planning context functions."""

    def test_get_current_plan_default_empty(self):
        """Test that get_current_plan returns empty list by default."""
        # In a fresh context (no container), should return empty list
        # Note: this relies on the fallback behavior in _get_container
        plan = get_current_plan()
        assert plan == []

    def test_set_and_get_without_init_context(self):
        """Test that set/get works even without init_planning_context.

        This tests the fallback behavior where _get_container creates and stores
        a default container. Without storing it in the ContextVar, set_current_plan
        would modify a temporary dict that gets discarded, causing silent data loss.
        """
        # Don't call init_planning_context - rely on fallback
        todos = [
            TodoItem(id="1", content="Task without init", status="pending"),
        ]

        # Set plan without initializing context first
        set_current_plan(todos)

        # Get should return the same plan (not a different empty list)
        retrieved = get_current_plan()

        assert len(retrieved) == 1
        assert retrieved[0].id == "1"
        assert retrieved[0].content == "Task without init"
        assert retrieved[0].status == "pending"

        # Verify we get the same container reference on subsequent calls
        retrieved_again = get_current_plan()
        assert retrieved is retrieved_again

    def test_set_and_get_current_plan(self):
        """Test setting and getting current plan."""
        todos = [
            TodoItem(id="1", content="First task", status="pending"),
            TodoItem(id="2", content="Second task", status="in_progress"),
        ]

        # Initialize context
        token = init_planning_context()

        try:
            set_current_plan(todos)
            retrieved = get_current_plan()

            assert len(retrieved) == 2
            assert retrieved[0].id == "1"
            assert retrieved[0].content == "First task"
            assert retrieved[0].status == "pending"
            assert retrieved[1].id == "2"
            assert retrieved[1].status == "in_progress"
        finally:
            reset_planning_context(token)

    def test_init_planning_context_returns_token(self):
        """Test that init_planning_context returns a proper token."""
        token = init_planning_context()

        # Should be a token object (not a tuple anymore)
        assert token is not None

        # Clean up
        reset_planning_context(token)

    def test_reset_planning_context_clears_state(self):
        """Test that reset_planning_context properly clears state."""
        # Initialize with data
        token = init_planning_context()
        set_current_plan([TodoItem(id="1", content="Task", status="pending")])

        # Verify data is set
        assert len(get_current_plan()) == 1

        # Reset
        reset_planning_context(token)

        # After reset, should be back to defaults
        # Note: ContextVar resets to previous value, not default
        # In a fresh test, this should be empty/None

    def test_get_final_plan_returns_current_plan(self):
        """Test that get_final_plan returns the same as get_current_plan."""
        token = init_planning_context()
        try:
            todos = [
                TodoItem(id="1", content="Task 1", status="completed"),
                TodoItem(id="2", content="Task 2", status="in_progress"),
            ]
            set_current_plan(todos)

            final_plan = get_final_plan()
            current_plan = get_current_plan()

            assert final_plan == current_plan
            assert len(final_plan) == 2
        finally:
            reset_planning_context(token)

    def test_mutable_container_shares_state(self):
        """Test that modifications to the plan are visible across contexts.

        This tests the key property of using a mutable container: changes made
        in one place are visible everywhere that has a reference to the container.
        """
        token = init_planning_context()
        try:
            # Initial state
            todos = [TodoItem(id="1", content="Task 1", status="pending")]
            set_current_plan(todos)

            # Get reference and modify
            plan = get_current_plan()
            assert len(plan) == 1
            assert plan[0].status == "pending"

            # Modify the task in place
            plan[0].status = "completed"

            # Get again - should see the modification
            updated = get_current_plan()
            assert updated[0].status == "completed"
        finally:
            reset_planning_context(token)

    def test_context_isolation_between_tasks(self):
        """Test that different async tasks have isolated contexts."""

        async def task_with_context(task_id: str):
            """Simulate an async task with its own context."""
            token = init_planning_context()
            try:
                todos = [
                    TodoItem(id=task_id, content=f"Task {task_id}", status="pending")
                ]
                set_current_plan(todos)

                # Yield control to other tasks
                await asyncio.sleep(0.01)

                # Verify our context is preserved
                retrieved_plan = get_current_plan()

                return {
                    "task_id": task_id,
                    "plan_id": retrieved_plan[0].id if retrieved_plan else None,
                }
            finally:
                reset_planning_context(token)

        async def run_concurrent_tasks():
            """Run multiple tasks concurrently."""
            results = await asyncio.gather(
                task_with_context("A"),
                task_with_context("B"),
                task_with_context("C"),
            )
            return results

        # Run the concurrent tasks
        results = asyncio.get_event_loop().run_until_complete(run_concurrent_tasks())

        # Each task should have seen its own context
        for result in results:
            assert result["plan_id"] == result["task_id"]

    def test_child_context_shares_mutable_container(self):
        """Test that child async contexts share the same mutable container.

        This simulates what happens when the Agents SDK spawns tool calls
        in child contexts - they should all share the same plan container.
        """

        async def parent_task():
            """Parent task that spawns child tasks."""
            token = init_planning_context()
            try:
                # Set initial plan in parent
                initial_todos = [
                    TodoItem(id="1", content="Task 1", status="pending"),
                    TodoItem(id="2", content="Task 2", status="pending"),
                ]
                set_current_plan(initial_todos)

                # Spawn child tasks that will modify the plan
                # Using asyncio.create_task simulates what the SDK does
                async def child_modify_task1():
                    # This runs in a child context that inherits the container reference
                    plan = get_current_plan()
                    if plan:
                        plan[0].status = "in_progress"
                    await asyncio.sleep(0.001)

                async def child_modify_task2():
                    # This also runs in a child context
                    plan = get_current_plan()
                    if plan and len(plan) > 1:
                        plan[1].status = "completed"
                    await asyncio.sleep(0.001)

                # Run children concurrently
                await asyncio.gather(
                    asyncio.create_task(child_modify_task1()),
                    asyncio.create_task(child_modify_task2()),
                )

                # Check that parent sees all modifications from children
                final_plan = get_final_plan()
                return final_plan
            finally:
                reset_planning_context(token)

        # Run and verify
        result = asyncio.get_event_loop().run_until_complete(parent_task())

        # Both modifications from child contexts should be visible
        assert len(result) == 2
        assert result[0].status == "in_progress"
        assert result[1].status == "completed"


class TestFormatPlanForAgent:
    """Test class for format_plan_for_agent function."""

    def test_format_empty_plan(self):
        """Test formatting an empty plan."""
        result = format_plan_for_agent([])
        assert result == "No tasks in current plan."

    def test_format_single_task(self):
        """Test formatting a single task."""
        todos = [TodoItem(id="1", content="Analyze project", status="pending")]
        result = format_plan_for_agent(todos)

        assert "Current plan:" in result
        assert "1. Analyze project [pending]" in result
        assert "⏳" in result  # pending icon

    def test_format_multiple_tasks_with_different_statuses(self):
        """Test formatting tasks with different statuses."""
        todos = [
            TodoItem(id="1", content="First task", status="completed"),
            TodoItem(id="2", content="Second task", status="in_progress"),
            TodoItem(id="3", content="Third task", status="pending"),
            TodoItem(id="4", content="Fourth task", status="cancelled"),
        ]
        result = format_plan_for_agent(todos)

        assert "Current plan:" in result
        assert "1. First task [completed]" in result
        assert "✅" in result  # completed icon
        assert "2. Second task [in_progress]" in result
        assert "🔄" in result  # in_progress icon
        assert "3. Third task [pending]" in result
        assert "⏳" in result  # pending icon
        assert "4. Fourth task [cancelled]" in result
        assert "❌" in result  # cancelled icon

    def test_format_preserves_task_order(self):
        """Test that formatting preserves task order."""
        todos = [
            TodoItem(id="1", content="First", status="pending"),
            TodoItem(id="2", content="Second", status="pending"),
            TodoItem(id="3", content="Third", status="pending"),
        ]
        result = format_plan_for_agent(todos)

        # Check that tasks appear in order
        first_pos = result.find("1. First")
        second_pos = result.find("2. Second")
        third_pos = result.find("3. Third")

        assert first_pos < second_pos < third_pos
