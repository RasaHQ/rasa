"""Planning tools for the Agents SDK copilot.

These tools allow the copilot to create and manage task plans that are
displayed to the user on the frontend. The copilot calls these tools to:

1. Create a plan before starting complex multi-step tasks
2. Update task status as work progresses

The tools use ContextVar for both plan storage and queue access to ensure
request isolation in concurrent scenarios. Plan is persisted to the database
only at the end of the request.
"""

import asyncio
from contextvars import ContextVar, Token
from typing import List, Optional

import structlog
from agents import function_tool

from rasa.builder.copilot.agent_sdk.tools.constants import (
    TOOL_CREATE_PLAN,
    TOOL_UPDATE_TASK,
)
from rasa.builder.copilot.agent_sdk.tools.planning_context import (
    format_plan_for_agent,
    get_current_plan,
    set_current_plan,
)
from rasa.builder.copilot.models import (
    TaskPlan,
    TaskStatus,
    TaskStatusUpdate,
    TodoItem,
    TodoPlanUpdate,
)
from rasa.builder.telemetry.langfuse.langfuse_compat import observe

structlogger = structlog.get_logger()


# ContextVar for request-scoped queue access
# This ensures each concurrent request uses its own queue, preventing
# plan updates from being sent to the wrong user's SSE stream.
_plan_queue_ctx: ContextVar[Optional[asyncio.Queue[TodoPlanUpdate]]] = ContextVar(
    "plan_queue_ctx", default=None
)


def set_plan_queue(
    queue: asyncio.Queue[TodoPlanUpdate],
) -> Token[Optional[asyncio.Queue[TodoPlanUpdate]]]:
    """Set the queue for sending plan updates to the response stream.

    Uses ContextVar to ensure request isolation - each concurrent request
    will have its own queue reference that won't be overwritten by other requests.

    Args:
        queue: The asyncio Queue to send TodoPlanUpdate events to.

    Returns:
        A token for use with reset_plan_queue to restore the previous state.
    """
    token = _plan_queue_ctx.set(queue)
    structlogger.debug(
        "planning_tools.set_plan_queue",
        queue_id=id(queue),
    )
    return token


def reset_plan_queue(token: Token[Optional[asyncio.Queue[TodoPlanUpdate]]]) -> None:
    """Reset the plan queue after a request completes.

    This should be called in a finally block to ensure cleanup.

    Args:
        token: Token returned from set_plan_queue.
    """
    _plan_queue_ctx.reset(token)
    structlogger.debug("planning_tools.reset_plan_queue")


def _send_plan_update_to_frontend(todos: List[TodoItem]) -> None:
    """Send the current plan state to the frontend queue.

    Uses ContextVar to get the queue for the current request context,
    ensuring updates go to the correct user's SSE stream.

    If the queue is full, the oldest update is dropped to make room for the
    newest one, since each update contains the full plan state.
    """
    plan_queue = _plan_queue_ctx.get()
    if plan_queue is None:
        structlogger.warning(
            "planning_tools.send_plan_update.no_queue",
            event_info="Plan queue not set, cannot send update",
        )
        return

    update = TodoPlanUpdate(tasks=[t.model_copy() for t in todos])
    try:
        plan_queue.put_nowait(update)
        structlogger.info(
            "planning_tools.send_plan_update.sent",
            task_count=len(todos),
            task_statuses=[t.status for t in todos],
        )
    except asyncio.QueueFull:
        # Drop oldest update to make room for the newest state
        plan_queue.get_nowait()
        plan_queue.put_nowait(update)
        structlogger.warning(
            "planning_tools.send_plan_update.dropped_oldest",
            event_info="Dropped oldest plan update to make room for newest",
            task_count=len(todos),
        )


@function_tool(name_override=TOOL_CREATE_PLAN)
@observe(name=f"planning_tool.{TOOL_CREATE_PLAN}", as_type="generation")
async def create_plan(plan: TaskPlan) -> str:
    """Create a task plan for a complex multi-step task.

    Use this tool when the user's request requires multiple distinct steps
    to complete. For example:
    - Creating a new flow (requires: create flow file, update domain,
      validate, train)
    - Adding a new slot with custom logic (e.g., define slot, add to
      domain, create action)
    - Debugging a complex issue (requires: analyze logs, identify cause,
      implement fix)

    Do NOT use this tool for:
    - Simple questions ("What is a slot?", "How do flows work?")
    - Single-step tasks ("Show me the domain file", "What flows exist?")
    - Quick explanations or clarifications

    Args:
        plan: The task plan with list of task descriptions.

    Returns:
        The created plan with all tasks and their status.
    """
    # Create TodoItems from the task descriptions
    todos = [
        TodoItem(id=str(i + 1), content=task, status=TaskStatus.PENDING)
        for i, task in enumerate(plan.tasks)
    ]

    # Store in context (ContextVar + fallback dict)
    set_current_plan(todos)

    structlogger.info(
        "planning_tools.create_plan",
        task_count=len(todos),
        tasks=[t.content for t in todos],
    )

    # Send to frontend via queue
    _send_plan_update_to_frontend(todos)

    # Return full plan to keep it fresh in agent's context
    return format_plan_for_agent(todos)


@function_tool(name_override=TOOL_UPDATE_TASK)
@observe(name=f"planning_tool.{TOOL_UPDATE_TASK}", as_type="generation")
async def update_task(update: TaskStatusUpdate) -> str:
    """Update the status of a task in the current plan.

    Call this tool to:
    - Mark a task as 'in_progress' when you start working on it
    - Mark a task as 'completed' when you finish it successfully
    - Mark a task as 'failed' when a task encounters an error that prevents
      completion (e.g., validation errors, missing dependencies, tool failures)
    - Mark a task as 'cancelled' if it's no longer needed

    After updating, this returns the full current plan state so you can
    see remaining tasks and stay on track.

    Args:
        update: The task status update with task ID and new status.

    Returns:
        The full current plan state with all tasks and their status.
    """
    # Get current plan (tries ContextVar first, then fallback dict)
    todos = get_current_plan()

    structlogger.debug(
        "planning_tools.update_task.start",
        task_id=update.task_id,
        new_status=update.status,
        plan_size=len(todos),
    )

    # Find and update the task
    task_found = False
    for task in todos:
        if task.id == update.task_id:
            old_status = task.status
            task.status = update.status
            task_found = True

            structlogger.info(
                "planning_tools.update_task.updated",
                task_id=update.task_id,
                old_status=old_status,
                new_status=update.status,
                task_content=task.content,
            )
            break

    if not task_found:
        structlogger.warning(
            "planning_tools.update_task.not_found",
            task_id=update.task_id,
            available_ids=[t.id for t in todos],
        )
        current_state = format_plan_for_agent(todos)
        return (
            f"Task with ID '{update.task_id}' not found in the current plan.\n\n"
            f"{current_state}"
        )

    # Update context with modified list
    set_current_plan(todos)

    # Send updated plan to frontend
    _send_plan_update_to_frontend(todos)

    # Return full plan state to keep it fresh in agent's context
    return format_plan_for_agent(todos)


# Export the function tools for use in the agent
PLANNING_TOOLS = [create_plan, update_task]
