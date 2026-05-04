"""Unified CLI for running evaluation experiments.

Usage:
    # To run offline eval on message classifier
    python -m rasa.builder.evaluator.run_experiment \
        --config rasa/builder/evaluator/configs/test_message_classifier.yaml
"""

import argparse
import sys

import structlog

from rasa.builder.evaluator.helpers import validate_env
from rasa.builder.evaluator.runner import ExperimentRunner

structlogger = structlog.get_logger()


def _run(config_path: str) -> int:
    structlogger.info("run_experiment.start", config_path=config_path)
    try:
        runner = ExperimentRunner(config_path=config_path)
        result = runner.run_experiment()
    except Exception as e:
        structlogger.error("run_experiment.failed", error=str(e))
        return 1

    structlogger.info(
        "run_experiment.completed",
        dataset_run_id=result.dataset_run_id,
        dataset_run_url=result.dataset_run_url,
    )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run an evaluation experiment defined by a YAML config file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config file.",
    )
    args = parser.parse_args()

    validate_env(push_langfuse=True)

    return _run(args.config)


if __name__ == "__main__":
    sys.exit(main())
