"""Shared utilities for Langfuse runners.

This module provides common functionality used across different Langfuse runner
implementations.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from rasa.builder.telemetry.langfuse_compat import require_langfuse

# Ensure langfuse is available - raises ImportError if not
require_langfuse()

from langfuse.experiment import ExperimentItem, ExperimentResult  # noqa: E402, TID251

from rasa.builder.evaluator.dataset.models import DatasetEntry  # noqa: E402
from rasa.builder.evaluator.shared.copilot_executor import (  # noqa: E402
    CopilotRunResult,
    run_copilot_with_response_handler,
)

structlogger = structlog.get_logger()


async def run_copilot_task(
    *,
    item: ExperimentItem,
    **kwargs: Dict[str, Any],
) -> Optional[CopilotRunResult]:
    """Copilot task function that processes each dataset item.

    Follows the langfuse.experiment.TaskFunction protocol. The function mimics the
    functionality of the `/copilot` endpoint.

    Args:
        item: The dataset item to process.
        kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the complete response and the generation context.
    """
    # Try to create the copilot context used for generating the response from the
    # dataset item, if the context cannot be created, skip the evaluation by
    # returning None.
    try:
        dataset_entry = DatasetEntry.from_raw_data(
            id=item.id,  # type: ignore[union-attr]
            input_data=item.input,  # type: ignore[union-attr]
            expected_output_data=item.expected_output,  # type: ignore[union-attr]
            metadata_data=item.metadata,  # type: ignore[union-attr]
        )
        context = dataset_entry.to_copilot_context()
    except Exception as e:
        structlogger.error(
            "langfuse_utils.run_copilot_task.context_creation_failed",
            event_info=(
                f"Failed to create CopilotContext from dataset item with id: "
                f"{item.id}. The Copilot cannot be run without a valid "  # type: ignore[union-attr]
                f"CopilotContext. Skipping evaluation."
            ),
            item_id=item.id,  # type: ignore[union-attr]
            item_input=item.input,  # type: ignore[union-attr]
            item_expected_output=item.expected_output,  # type: ignore[union-attr]
            error=str(e),
        )
        return None

    # Run the evaluation. If the task fails, skip the evaluation by returning None.
    try:
        return await run_copilot_with_response_handler(context)
    except Exception as e:
        structlogger.error(
            "langfuse_utils.run_copilot_task.copilot_run_failed",
            event_info=(
                f"Failed to run the copilot with response handler for dataset item "
                f"with id: {item.id}. Skipping evaluation."  # type: ignore[union-attr]
            ),
            item_id=item.id,  # type: ignore[union-attr]
            item_input=item.input,  # type: ignore[union-attr]
            item_expected_output=item.expected_output,  # type: ignore[union-attr]
            error=str(e),
        )
        return None


def report_langfuse_run_results_to_txt_file(
    result: ExperimentResult, output_path: Path
) -> None:
    result_str = result.format().replace("\\n", "\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(output_path), "w") as f:
        f.write(result_str)
    structlogger.info(
        "langfuse_runner._report_run_results.exported",
        event_info="Evaluation results exported to text file",
        text_file=output_path,
    )
