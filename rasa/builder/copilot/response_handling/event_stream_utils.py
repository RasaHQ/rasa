"""Utilities for concurrent event stream processing.

This module provides helper functions for managing asyncio tasks that monitor
multiple event sources (LLM streams, queues) concurrently.
"""

import asyncio
from typing import AsyncGenerator, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import structlog
from agents import StreamEvent

from rasa.builder.copilot.models import MCPToolCall, TaskType, TodoPlanUpdate

structlogger = structlog.get_logger()

T = TypeVar("T")


async def cancel_task_safely(task: Optional[asyncio.Task]) -> None:
    """Cancel a task and await its completion safely.

    Args:
        task: The task to cancel, or None (no-op).
    """
    if task is None:
        return

    if task.done():
        structlogger.debug(
            "event_stream_utils.cancel_task_safely.already_done",
            event_info="Task already done, skipping cancel",
        )
        return

    task.cancel()

    # Use asyncio.wait() instead of await to avoid catching CancelledError directly.
    # asyncio.wait() doesn't propagate task exceptions; we inspect the task state after.
    await asyncio.wait([task])

    if task.cancelled():
        structlogger.debug(
            "event_stream_utils.cancel_task_safely.cancelled",
            event_info="Task was cancelled (expected)",
        )
    else:
        # Task completed (possibly with an exception) despite cancellation request
        try:
            task.result()
            structlogger.debug(
                "event_stream_utils.cancel_task_safely.completed",
                event_info="Task completed after cancel",
            )
        except StopAsyncIteration:
            structlogger.debug(
                "event_stream_utils.cancel_task_safely.stop_async_iteration",
                event_info="Stream stopped (expected)",
            )
        except Exception:
            structlogger.exception(
                "event_stream_utils.cancel_task_safely.error",
                event_info="Task raised an unexpected exception after cancel",
            )


async def drain_done_task(
    task: Optional[asyncio.Task[T]],
    on_result: Optional[Callable[[T], None]] = None,
) -> Tuple[Optional[T], bool]:
    """Extract result from a done task, cancelling if not done.

    Args:
        task: The task to drain.
        on_result: Optional callback to process the result.

    Returns:
        Tuple of (result or None, was_successful).
    """
    if task is None:
        return None, False

    if task.done():
        # Check task.cancelled() before task.result() to avoid catching CancelledError
        if task.cancelled():
            structlogger.debug(
                "event_stream_utils.drain_done_task.cancelled",
                task_done=True,
                event_info="Task was cancelled",
            )
            return None, False

        try:
            result = task.result()
            if on_result:
                on_result(result)
            structlogger.debug(
                "event_stream_utils.drain_done_task.completed",
                task_done=True,
                has_result=result is not None,
            )
            return result, True
        except Exception:
            structlogger.exception(
                "event_stream_utils.drain_done_task.error",
                event_info="Done task raised an exception when getting result",
            )
            return None, False
    else:
        structlogger.debug(
            "event_stream_utils.drain_done_task.cancelling",
            task_done=False,
            event_info="Task not done, cancelling before drain",
        )
        await cancel_task_safely(task)
        return None, False


class TaskManager:
    """Manages asyncio tasks for concurrent stream/queue monitoring.

    This class encapsulates the task lifecycle management for monitoring
    multiple event sources (LLM stream, MCP queue, plan queue) concurrently.
    """

    def __init__(
        self,
        stream_iterator: AsyncGenerator[StreamEvent, None],
        mcp_queue: Optional[asyncio.Queue[MCPToolCall]],
        plan_queue: Optional[asyncio.Queue[TodoPlanUpdate]],
        process_mcp_event: Callable[[MCPToolCall], None],
        process_plan_event: Callable[[TodoPlanUpdate], None],
    ) -> None:
        """Initialize the task manager.

        Args:
            stream_iterator: Async iterator for the LLM stream.
            mcp_queue: Queue for MCP tool call events.
            plan_queue: Queue for plan update events.
            process_mcp_event: Callback to process MCP events.
            process_plan_event: Callback to process plan events.
        """
        self._stream_iterator = stream_iterator
        self._mcp_queue = mcp_queue
        self._plan_queue = plan_queue
        self._process_mcp_event = process_mcp_event
        self._process_plan_event = process_plan_event

        self._stream_task: Optional[asyncio.Task] = None
        self._mcp_task: Optional[asyncio.Task] = None
        self._plan_task: Optional[asyncio.Task] = None
        self._stream_exhausted = False

    @property
    def stream_exhausted(self) -> bool:
        return self._stream_exhausted

    def create_tasks_if_needed(
        self,
    ) -> Tuple[List[asyncio.Task], Dict[asyncio.Task, TaskType]]:
        """Create tasks for stream and queues if needed.

        Returns:
            Tuple of (list of active tasks, dict mapping tasks to their types).
        """
        tasks: List[asyncio.Task] = []
        task_types: Dict[asyncio.Task, TaskType] = {}

        # Create stream task if stream is not exhausted.
        if not self._stream_exhausted and self._stream_task is None:
            self._stream_task = asyncio.create_task(anext(self._stream_iterator))

        if self._stream_task is not None:
            tasks.append(self._stream_task)
            task_types[self._stream_task] = TaskType.STREAM

        # Create MCP queue task
        if self._mcp_queue is not None and self._mcp_task is None:
            self._mcp_task = asyncio.create_task(self._mcp_queue.get())

        if self._mcp_task is not None:
            tasks.append(self._mcp_task)
            task_types[self._mcp_task] = TaskType.MCP

        # Create plan queue task
        if self._plan_queue is not None and self._plan_task is None:
            self._plan_task = asyncio.create_task(self._plan_queue.get())

        if self._plan_task is not None:
            tasks.append(self._plan_task)
            task_types[self._plan_task] = TaskType.PLAN

        return tasks, task_types

    def process_completed_task(
        self, task: asyncio.Task, task_type: TaskType
    ) -> Optional[Union[StreamEvent, MCPToolCall, TodoPlanUpdate]]:
        """Process a completed task and return its result.

        Args:
            task: The completed task.
            task_type: Type of task (STREAM, MCP, or PLAN).

        Returns:
            The task result, or None if the stream was exhausted or the task failed.
        """
        try:
            result = task.result()

            if task_type == TaskType.STREAM:
                self._stream_task = None
                return result
            elif task_type == TaskType.MCP:
                self._mcp_task = None
                self._process_mcp_event(result)
                return result
            elif task_type == TaskType.PLAN:
                self._plan_task = None
                self._process_plan_event(result)
                return result

        except StopAsyncIteration:
            self._stream_exhausted = True
            self._stream_task = None
            structlogger.debug(
                "event_stream_utils.process_completed_task.stream_exhausted",
                event_info="LLM response stream exhausted",
            )

        return None

    async def drain_remaining_queue_events(
        self,
    ) -> List[Union[MCPToolCall, TodoPlanUpdate]]:
        """Drain remaining queue events when stream is exhausted.

        Returns:
            List of events that were ready in the queues.
        """
        events: List[Union[MCPToolCall, TodoPlanUpdate]] = []

        # Process MCP task
        result, success = await drain_done_task(self._mcp_task, self._process_mcp_event)
        if success and result is not None:
            events.append(result)
        self._mcp_task = None

        # Process plan task
        result, success = await drain_done_task(
            self._plan_task, self._process_plan_event
        )
        if success and result is not None:
            events.append(result)
        self._plan_task = None

        return events

    async def cleanup(self) -> None:
        """Cancel all pending tasks."""
        await cancel_task_safely(self._stream_task)
        await cancel_task_safely(self._mcp_task)
        await cancel_task_safely(self._plan_task)
