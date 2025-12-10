import os
import sys
from typing import Any, List, Optional

import structlog

from rasa.builder.telemetry.langfuse_compat import require_langfuse

# Ensure langfuse is available - raises ImportError if not
require_langfuse()

from langfuse.experiment import ExperimentResult  # noqa: E402, TID251

structlogger = structlog.get_logger()


def validate_environment(required_env_vars: List[str]) -> None:
    """Validate that all required environment variables are set."""
    missing_vars: List[str] = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        structlogger.error(
            "scipts.utils.validate_environment.missing_variables",
            event_info=(
                "Missing required environment variables. Please set the following "
                f"environment variables: {missing_vars}."
            ),
            missing_variables=missing_vars,
        )
        sys.exit(1)


async def run_experiment(runner: Any, dataset_name: str) -> Optional[ExperimentResult]:
    """Run the evaluation experiment on the given runner."""
    try:
        structlogger.info(
            "scripts.utils.run_experiment.starting",
            event_info=(
                f"Starting evaluation experiment on the "
                f"{runner.__class__.__name__} runner."
            ),
            dataset_name=dataset_name,
        )

        # Initialize and run the experiment
        result = runner.run_experiment()

        structlogger.info(
            "scripts.utils.run_experiment.completed",
            event_info=(
                f"Evaluation experiment on the {runner.__class__.__name__} "
                f"runner completed successfully.",
            ),
            dataset_name=dataset_name,
        )
        return result

    except Exception as e:
        structlogger.error(
            "scripts.utils.run_experiment.failed",
            event_info=(
                f"Evaluation experiment on the {runner.__class__.__name__} "
                f"runner failed."
            ),
            error=str(e),
            dataset_name=dataset_name,
        )
        return None
