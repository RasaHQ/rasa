#!/usr/bin/env python3
"""Response Classification Evaluator CLI.

A command-line tool for running response classification evaluation experiments using
Langfuse.

This script runs experiments on datasets and provides links to the results.
"""

import argparse
import asyncio
import os
import sys
from typing import Optional

import structlog
from langfuse.experiment import ExperimentResult

from rasa.builder.evaluator.constants import (
    DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME,
    RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME,
)
from rasa.builder.evaluator.response_classification.langfuse_runner import (
    ResponseClassificationLangfuseRunner,
)

# Configure structured logging
structlogger = structlog.get_logger()


def validate_environment() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "INKEEP_API_KEY",
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        structlogger.error(
            "main.validate_environment.missing_variables",
            event_info=(
                "Missing required environment variables. Please set the following "
                f"environment variables: {missing_vars}."
            ),
            missing_variables=missing_vars,
        )
        sys.exit(1)


async def run_experiment(
    dataset_name: str, output_file: Optional[str] = None
) -> Optional[ExperimentResult]:
    """Run the response classification evaluation experiment."""
    try:
        structlogger.info(
            "main.run_experiment.starting",
            event_info="Starting response classification evaluation experiment",
            dataset_name=dataset_name,
        )

        # Initialize and run the experiment
        runner = ResponseClassificationLangfuseRunner(
            dataset_name=dataset_name, output_dir=output_file
        )
        result = runner.run_experiment()

        structlogger.info(
            "main.run_experiment.completed",
            event_info=(
                "Response classification evaluation experiment completed successfully",
            ),
            dataset_name=dataset_name,
        )

        structlogger.info("✅ Experiment completed successfully!")

        return result

    except Exception as e:
        structlogger.error(
            "main.run_experiment.failed",
            event_info="Response classification evaluation experiment failed",
            error=str(e),
            dataset_name=dataset_name,
        )
        structlogger.error(f"❌ Error running experiment: {e}")
        return None


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run response classification evaluation experiments using Langfuse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\npython run_response_classification_evaluator.py my_dataset",
    )

    parser.add_argument(
        "--dataset-name",
        help="Name of the dataset to evaluate",
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help=(
            "(Optional) Directory to write experiment results. Two files are created "
            "with a timestamp prefix (YYYYMMDD_HHMMSS_):\n"
            f"- {DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME} "
            f"from Langfuse, and"
            f"- {RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME} "
            "from the classifier."
        ),
    )

    args = parser.parse_args()

    # Validate environment variables
    validate_environment()

    structlogger.info(f"🔍 Dataset: {args.dataset_name}")
    structlogger.info("🚀 Starting evaluation...")

    # Run the experiment
    result = asyncio.run(run_experiment(args.dataset_name, args.output_file))

    if result is None:
        sys.exit(1)

    # Get experiment link
    structlogger.info(
        "✨ Evaluation complete!",
        dataset_run_id=result.dataset_run_id,
        dataset_run_url=result.dataset_run_url,
    )

    # Print formatted results:
    result_str = result.format().replace("\\n", "\n")
    structlogger.info(result_str)

    return 0


if __name__ == "__main__":
    main()
