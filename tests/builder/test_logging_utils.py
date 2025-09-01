"""Tests for logging utilities."""

import asyncio
import threading
from typing import Any, Dict, List

import pytest
import structlog

from rasa.builder.logging_utils import capture_validation_logs


class TestValidationLogCapture:
    """Test cases for validation log capture functionality."""

    @staticmethod
    async def async_worker(task_id: int, results: Dict[int, Dict[str, Any]]) -> None:
        """Worker function for each async task."""
        with capture_validation_logs() as captured_logs:
            structlogger = structlog.get_logger()
            structlogger.error(
                f"task.{task_id}.error", event_info=f"Error from task {task_id}"
            )
            structlogger.info(
                f"task.{task_id}.info", event_info=f"Info from task {task_id}"
            )
            structlogger.debug(
                f"task.{task_id}.debug",
                event_info=f"Debug from task {task_id}",
            )
            structlogger.warning(
                f"task.{task_id}.warning",
                event_info=f"Warning from task {task_id}",
            )
            structlogger.critical(
                f"task.{task_id}.critical",
                event_info=f"Critical from task {task_id}",
            )

            # Store results for this task
            results[task_id] = {
                "count": len(captured_logs),
                "events": [log["event"] for log in captured_logs],
                "task_logs": captured_logs.copy(),
            }

    @staticmethod
    def thread_worker(thread_id: int, results: Dict[int, Dict[str, Any]]) -> None:
        """Worker function for each thread."""
        with capture_validation_logs() as captured_logs:
            structlogger = structlog.get_logger()
            structlogger.error(
                f"thread.{thread_id}.error",
                event_info=f"Error from thread {thread_id}",
            )
            structlogger.info(
                f"thread.{thread_id}.info",
                event_info=f"Info from thread {thread_id}",
            )
            structlogger.debug(
                f"thread.{thread_id}.debug",
                event_info=f"Debug from thread {thread_id}",
            )
            structlogger.warning(
                f"thread.{thread_id}.warning",
                event_info=f"Warning from thread {thread_id}",
            )
            structlogger.critical(
                f"thread.{thread_id}.critical",
                event_info=f"Critical from thread {thread_id}",
            )

            # Store results for this thread
            results[thread_id] = {
                "count": len(captured_logs),
                "events": [log["event"] for log in captured_logs],
                "thread_logs": captured_logs.copy(),
            }

    @pytest.mark.asyncio
    async def test_capture_validation_logs_async_isolation(self):
        """Test that logs are isolated between different async tasks."""
        # Given
        results: Dict[int, Dict[str, Any]] = {}
        num_tasks = 10

        # When
        tasks: List[asyncio.Task[None]] = []
        for i in range(num_tasks):
            task = asyncio.create_task(self.async_worker(i, results))
            tasks.append(task)
        await asyncio.gather(*tasks)

        # Then
        for task_id in range(num_tasks):
            # Verify each task captured only its own logs
            task_result = results[task_id]
            assert task_result["count"] == 5
            for log_level in ["error", "info", "debug", "warning", "critical"]:
                assert f"task.{task_id}.{log_level}" in task_result["events"]

            # Verify no cross-contamination between tasks
            task_logs = results[task_id]["task_logs"]
            for log in task_logs:
                assert log["event"].startswith(f"task.{task_id}.")

    def test_capture_validation_logs_thread_isolation(self):
        """Test that logs are isolated between different threads."""
        # Given
        results: Dict[int, Dict[str, Any]] = {}
        num_threads = 10

        # When
        for i in range(num_threads):
            thread = threading.Thread(target=self.thread_worker, args=(i, results))
            thread.start()
            thread.join()

        # Then
        for thread_id in range(num_threads):
            # Verify each thread captured only its own logs
            thread_result = results[thread_id]
            assert thread_result["count"] == 5
            for log_level in ["error", "info", "debug", "warning", "critical"]:
                assert f"thread.{thread_id}.{log_level}" in thread_result["events"]

            # Verify no cross-contamination between threads
            thread_logs = results[thread_id]["thread_logs"]
            for log in thread_logs:
                assert log["event"].startswith(f"thread.{thread_id}.")

    def test_capture_validation_logs_cleanup(self):
        """Test that logs are properly cleaned up after context manager exits."""
        # First capture
        with capture_validation_logs() as captured_logs:
            structlogger = structlog.get_logger()
            structlogger.error("test.error", event_info="Error message")
            assert len(captured_logs) == 1

        # Second capture should be empty (clean slate)
        with capture_validation_logs() as captured_logs:
            assert len(captured_logs) == 0
