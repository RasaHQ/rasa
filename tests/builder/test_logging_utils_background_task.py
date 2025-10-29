"""Test validation log capture in background task scenarios."""

import asyncio
from typing import Any, Dict

import pytest
import structlog

from rasa.builder.logging_utils import (
    capture_validation_logs,
    collecting_validation_logs_processor,
)


@pytest.fixture(scope="function", autouse=True)
def configure_structlog_for_tests():
    """Ensure structlog is configured with our validation log processor for tests."""
    original_config = structlog.get_config()
    original_processors = original_config["processors"]

    # Add our processor if it's not already there
    if collecting_validation_logs_processor not in original_processors:
        new_processors = [collecting_validation_logs_processor] + original_processors
        structlog.configure(processors=new_processors)

    yield

    # Restore original config
    structlog.configure(processors=original_processors)


class TestValidationLogCaptureBackgroundTask:
    """Test cases simulating the background task scenario from service.py."""

    @staticmethod
    async def simulated_validation_in_background_task(
        captured_logs_container: dict,
    ) -> None:
        """Simulate validation running in a background task.

        Simulates run_replace_all_files_job. This simulates the actual flow:
        1. Context manager is set up in main function
        2. Background task is created with app.add_task()
        3. Validation runs and logs errors
        4. Logs should be captured in the main function's context
        """
        with capture_validation_logs() as captured_logs:
            # This simulates what happens in jobs.py -> validate_project()
            # which calls validate_files() that logs errors
            structlogger = structlog.get_logger()

            # Simulate various validation logs
            structlogger.info(
                "validation.flows.started",
                event_info="Starting flow validation",
            )

            # This is the critical error that was missing in production
            structlogger.error(
                "validator.verify_predicates.link.invalid_condition",
                step="react_to_mood",
                link='slots.user_mood in ["sad", "stressed", "angry"]',
                flow="mood_check",
                event_info=(
                    "Detected invalid condition 'slots.user_mood in "
                    '["sad", "stressed", "angry"]\' at step \'react_to_mood\' '
                    "for flow id 'mood_check'. Please make sure that all "
                    "conditions are valid."
                ),
            )

            structlogger.info(
                "validation.flows.ended",
                event_info="Flow validation completed",
            )

            structlogger.error(
                "cli.validate_files.project_validation_error",
                event_info="Project validation completed with errors.",
            )

            # Store the captured logs in the container
            captured_logs_container["logs"] = captured_logs.copy()

    @pytest.mark.asyncio
    async def test_validation_log_capture_in_background_task(self) -> None:
        """Test validation logs captured in background task.

        This test simulates the actual production scenario:
        - Frontend calls POST /files
        - Service creates background task with request.app.add_task()
        - Background task runs validation
        - Validation logs should be captured and included in error response
        """
        # Container to store results from background task
        captured_logs_container: Dict[str, Any] = {}

        # Simulate creating a background task (like request.app.add_task())
        task = asyncio.create_task(
            self.simulated_validation_in_background_task(captured_logs_container)
        )

        # Wait for the background task to complete
        await task

        # Verify logs were captured
        captured_logs = captured_logs_container["logs"]

        # Should have captured all 4 log entries
        assert (
            len(captured_logs) == 4
        ), f"Expected 4 log entries, got {len(captured_logs)}: {captured_logs}"

        # Verify the critical error was captured
        error_logs = [log for log in captured_logs if log["log_level"] == "error"]
        assert (
            len(error_logs) == 2
        ), f"Expected 2 error logs, got {len(error_logs)}: {error_logs}"

        # Verify the specific validation error is present
        validation_errors = [
            log
            for log in error_logs
            if log.get("event") == "validator.verify_predicates.link.invalid_condition"
        ]
        assert (
            len(validation_errors) == 1
        ), "Missing validator.verify_predicates.link.invalid_condition error"

        # Verify error details
        validation_error = validation_errors[0]
        assert validation_error["step"] == "react_to_mood"
        assert validation_error["flow"] == "mood_check"
        assert "invalid condition" in validation_error["event_info"]

    @pytest.mark.asyncio
    async def test_multiple_background_tasks_isolation(self) -> None:
        """Test multiple concurrent background tasks have isolated log capture."""

        async def task_with_logs(task_id: int, results: dict) -> None:
            """Background task that captures its own logs."""
            with capture_validation_logs() as captured_logs:
                structlogger = structlog.get_logger()
                structlogger.error(
                    f"task.{task_id}.error",
                    event_info=f"Error from task {task_id}",
                )
                results[task_id] = captured_logs.copy()

        # Run multiple tasks concurrently (like multiple API requests)
        results: Dict[int, Any] = {}
        tasks = [asyncio.create_task(task_with_logs(i, results)) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify each task captured only its own logs
        for task_id in range(5):
            assert len(results[task_id]) == 1
            assert results[task_id][0]["event"] == f"task.{task_id}.error"
