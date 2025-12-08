#!/usr/bin/env python3
"""Copilot Response Evaluator CLI.

A command-line tool for running copilot response quality evaluation experiments using
Langfuse.

This script evaluates copilot responses on faithfulness and completeness dimensions.
"""

import argparse
import asyncio
import sys

import structlog

from rasa.builder.evaluator.copilot_response_evaluator.langfuse_runner import (
    CopilotResponseEvaluatorLangfuseRunner,
)
from rasa.builder.evaluator.scripts.utils import run_experiment, validate_environment
from rasa.builder.evaluator.shared.constants import (
    COPILOT_RESPONSE_EVALUATION_YAML_OUTPUT_FILENAME,
    DEFAULT_COPILOT_RESPONSE_EVALUATION_TEXT_OUTPUT_FILENAME,
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
        description=(
            "Run copilot response quality evaluation experiments using Langfuse. "
            "Evaluates responses on faithfulness and completeness dimensions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "python run_copilot_response_evaluator.py --dataset-name my_dataset"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        help=(
            "Name of the dataset on the Langfuse platform to evaluate the copilot "
            "responses on."
        ),
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help=(
            "(Optional) Directory to write experiment results. Two files are created "
            "with a timestamp prefix (YYYYMMDD_HHMMSS_):\n"
            f"- {DEFAULT_COPILOT_RESPONSE_EVALUATION_TEXT_OUTPUT_FILENAME} "
            f"from Langfuse, and\n"
            f"- {COPILOT_RESPONSE_EVALUATION_YAML_OUTPUT_FILENAME} "
            "with detailed metrics per item."
        ),
    )

    args = parser.parse_args()

    # Validate environment variables
    validate_environment(REQUIRED_ENV_VARS)

    structlogger.info(f"🔍 Dataset: {args.dataset_name}")
    structlogger.info("🚀 Starting evaluation...")

    runner = CopilotResponseEvaluatorLangfuseRunner(
        dataset_name=args.dataset_name, output_dir=args.output_file
    )

    # Run the experiment
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
