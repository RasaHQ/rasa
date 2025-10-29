"""Functions for validating Rasa projects."""

import sys
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import structlog

from rasa.builder import config
from rasa.builder.exceptions import ValidationError
from rasa.builder.logging_utils import capture_validation_logs
from rasa.cli.validation.bot_config import validate_files
from rasa.shared.importers.importer import TrainingDataImporter

structlogger = structlog.get_logger()


@contextmanager
def _mock_sys_exit() -> Generator[Dict[str, bool], Any, None]:
    """Context manager to prevent sys.exit from being called during validation."""
    was_sys_exit_called = {"value": False}

    def sys_exit_mock(code: int = 0) -> None:
        was_sys_exit_called["value"] = True

    original_exit = sys.exit
    sys.exit = sys_exit_mock  # type: ignore

    try:
        yield was_sys_exit_called
    finally:
        sys.exit = original_exit


async def validate_project(importer: TrainingDataImporter) -> Optional[str]:
    """Validate a Rasa project.

    Args:
        importer: Training data importer with domain, flows, and config

    Returns:
        None if validation passes, error message if validation fails.

    Raises:
        ValidationError: If validation fails
    """
    with capture_validation_logs() as captured_logs:
        try:
            with _mock_sys_exit() as exit_tracker:
                from rasa.core.config.configuration import Configuration

                Configuration.initialise_empty()
                Configuration.initialise_sub_agents(sub_agents_path=None)

                validate_files(
                    fail_on_warnings=config.VALIDATION_FAIL_ON_WARNINGS,
                    max_history=config.VALIDATION_MAX_HISTORY,
                    importer=importer,
                )

                if exit_tracker["value"]:
                    error_logs = [
                        log for log in captured_logs if log.get("log_level") != "debug"
                    ]
                    structlogger.error(
                        "validation.failed.sys_exit",
                        error_logs=error_logs,
                    )
                    raise ValidationError(
                        "Validation failed with sys.exit", validation_logs=error_logs
                    )

                structlogger.info("validation.success")
                return None

        except ValidationError:
            raise

        except Exception as e:
            error_msg = f"Validation failed with exception: {e}"

            error_logs = [
                log for log in captured_logs if log.get("log_level") != "debug"
            ]

            structlogger.error(
                "validation.failed.exception", error=str(e), validation_logs=error_logs
            )
            raise ValidationError(error_msg, validation_logs=error_logs)

        except SystemExit as e:
            error_logs = [
                log for log in captured_logs if log.get("log_level") != "debug"
            ]

            structlogger.error(
                "validation.failed.sys_exit",
                error_logs=error_logs,
            )
            raise ValidationError(
                f"SystemExit during validation: {e}", validation_logs=error_logs
            )
