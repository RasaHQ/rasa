"""ContextVar-based storage for TODO planning state.

This module provides async-safe request-scoped storage for the copilot's
task planning state. Using ContextVar ensures that concurrent requests
don't share plan state, and enables access from tools, hooks, and middleware.

Usage:
    # At request start
    token = init_planning_context()

    # In tools - read/write plan
    plan = get_current_plan()
    set_current_plan(updated_plan)

    # At request end - get final plan for persistence
    final_plan = get_final_plan()
    reset_planning_context(token)
"""

from contextvars import ContextVar, Token
from typing import Any, Dict, List, Optional

import structlog

from rasa.builder.copilot.models import TodoItem

structlogger = structlog.get_logger()

# Key used for storing todos in the plan container
TODOS_KEY = "todos"

PlanContainer = Dict[str, Any]


def _todos_to_dicts(todos: List["TodoItem"]) -> List[Dict[str, Any]]:
    """Convert todos to dicts for logging."""
    return [todo.model_dump() for todo in todos]


# ContextVar holding a mutable container for plan state
# The container has key: TODOS_KEY (List[TodoItem])
# Using a mutable container ensures changes are visible across child async contexts
_plan_container_ctx: ContextVar[Optional[PlanContainer]] = ContextVar(
    "plan_container_ctx", default=None
)


def _get_container() -> PlanContainer:
    """Get the current plan container.

    Returns:
        The mutable container dict. Creates and stores a default if none exists
        (shouldn't happen if init_planning_context was called).
    """
    container = _plan_container_ctx.get()
    if container is None:
        structlogger.warning(
            "planning_context._get_container.no_container",
            event_info="No plan container initialized, creating default",
        )
        container = {TODOS_KEY: []}
        _plan_container_ctx.set(container)
    return container


def get_current_plan() -> List[TodoItem]:
    """Get the current TODO plan from context.

    Returns:
        List of TodoItem objects representing the current plan.
        Returns empty list if no plan has been created.
    """
    container = _get_container()
    return container.get(TODOS_KEY, [])


def set_current_plan(todos: List[TodoItem]) -> None:
    """Set the current TODO plan in context.

    This modifies the container in-place so changes are visible
    across all async contexts sharing this container.

    Args:
        todos: List of TodoItem objects to set as the current plan.
    """
    container = _get_container()
    previous_todos = container.get(TODOS_KEY, [])
    container[TODOS_KEY] = todos
    structlogger.debug(
        "planning_context.set_current_plan",
        before=_todos_to_dicts(previous_todos),
        after=_todos_to_dicts(todos),
    )


def get_final_plan() -> List[TodoItem]:
    """Get the final plan state for persistence at end of request.

    Returns:
        List of TodoItem objects, or empty list if no plan exists.
    """
    return get_current_plan()


def init_planning_context() -> Token[Optional[PlanContainer]]:
    """Initialize the planning context for a new request.

    This creates a new mutable container that will be shared across all
    child async contexts (tool calls). Returns a token for cleanup.

    Returns:
        A token for use with reset_planning_context.
    """
    # Create a fresh mutable container for this request
    container: PlanContainer = {TODOS_KEY: []}
    token = _plan_container_ctx.set(container)
    structlogger.debug("planning_context.init")
    return token


def reset_planning_context(
    token: Token[Optional[PlanContainer]],
) -> None:
    """Reset the planning context after a request completes.

    This should be called in a finally block to ensure cleanup.

    Args:
        token: Token returned from init_planning_context.
    """
    # Capture state before reset for debugging
    container = _plan_container_ctx.get()
    plan_before = _todos_to_dicts(container.get(TODOS_KEY, [])) if container else None
    _plan_container_ctx.reset(token)
    structlogger.debug(
        "planning_context.reset",
        plan_before_reset=plan_before,
    )


def format_plan_for_agent(todos: List[TodoItem]) -> str:
    """Format the TODO plan for inclusion in tool response.

    This keeps the plan "fresh" in the agent's context window by
    returning the full state after each update.

    Args:
        todos: List of TodoItem objects.

    Returns:
        Formatted string representation of the plan.
    """
    if not todos:
        return "No tasks in current plan."

    status_icons = {
        "pending": "⏳",
        "in_progress": "🔄",
        "completed": "✅",
        "cancelled": "❌",
    }

    lines = ["Current plan:"]
    for task in todos:
        icon = status_icons.get(task.status, "•")
        lines.append(f"  {task.id}. {task.content} [{task.status}] {icon}")

    return "\n".join(lines)
