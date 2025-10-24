"""Constants for the evaluator module."""

from pathlib import Path

# Base directory for the rasa package
BASE_DIR = Path(__file__).parent.parent.parent

# Response classification evaluation results directory
RESPONSE_CLASSIFICATION_EVALUATION_RESULTS_DIR = (
    BASE_DIR / "builder" / "evaluator" / "results"
)
# Default output filename
DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME = "run_results.txt"
# Default YAML output filename
RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME = "run_results.yaml"
