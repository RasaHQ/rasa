"""Constants for the evaluator module."""

from pathlib import Path

# Base directory for the rasa package
BASE_DIR = Path(__file__).parent.parent.parent

EVALUATOR_DIR = BASE_DIR / "builder" / "evaluator"

# Prompt templates directory and paths for the evaluator
CLAIM_EXTRACTOR_PROMPTS_PACKAGE_NAME = "builder.evaluator.content_processors.prompts"
CLAIM_EXTRACTOR_PROMPTS_FILE = "claim_extractor_prompt.jinja2"
CLAIM_EXTRACTOR_RESPONSE_SCHEMA_PATH = (
    EVALUATOR_DIR
    / "content_processors"
    / "prompts"
    / "claim_extractor_response_schema.json"
)

# Response classification evaluation results directory
RESPONSE_CLASSIFICATION_EVALUATION_RESULTS_DIR = (
    BASE_DIR / "builder" / "evaluator" / "results"
)
# Default output filename
DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME = "run_results.txt"
# Default YAML output filename
RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME = "run_results.yaml"
