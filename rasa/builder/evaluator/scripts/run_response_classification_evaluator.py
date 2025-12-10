#!/usr/bin/env python3
"""Response Classification Evaluator CLI.

A command-line tool for running response classification evaluation experiments using
Langfuse.

This script runs experiments on datasets and provides links to the results.
"""

import argparse
import asyncio
import sys

import structlog

from rasa.builder.evaluator.response_classification.langfuse_runner import (
    ResponseClassificationLangfuseRunner,
)
from rasa.builder.evaluator.scripts.utils import run_experiment, validate_environment
from rasa.builder.evaluator.shared.constants import (
    DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME,
    RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME,
)

# Configure structured logging
structlogger = structlog.get_logger()


REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "INKEEP_API_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run response classification evaluation experiments using Langfuse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\npython run_response_classification_evaluator.py my_dataset",
    )

    parser.add_argument(
        "--dataset-name",
        help=(
            "Name of the dataset on the Langfuse platform to evaluate the response "
            "classification on."
        ),
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
    validate_environment(REQUIRED_ENV_VARS)
    structlogger.info(f"🔍 Dataset: {args.dataset_name}")
    structlogger.info("🚀 Starting evaluation...")

    # Run the experiment
    runner = ResponseClassificationLangfuseRunner(
        dataset_name=args.dataset_name, output_dir=args.output_file
    )
    result = asyncio.run(run_experiment(runner, args.dataset_name))

    if result is None:
        structlogger.error("❌ Evaluation failed")
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
