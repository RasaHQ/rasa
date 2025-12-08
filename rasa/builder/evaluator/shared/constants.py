"""Constants for the evaluator module."""

from pathlib import Path

# Base directory for the rasa package
BASE_DIR = Path(__file__).parent.parent.parent.parent

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

# Prompt templates directory and paths for the faithfulness judge
FAITHFULNESS_JUDGE_PROMPTS_PACKAGE_NAME = "builder.evaluator.faithfulness_judge.prompts"
FAITHFULNESS_JUDGE_PROMPTS_FILE = "faithfulness_judge_prompt.jinja2"
FAITHFULNESS_JUDGE_RESPONSE_SCHEMA_PATH = (
    EVALUATOR_DIR
    / "faithfulness_judge"
    / "prompts"
    / "faithfulness_judge_response_schema.json"
)

# Prompt templates directory and paths for the completeness judge
COMPLETENESS_JUDGE_PROMPTS_PACKAGE_NAME = "builder.evaluator.completeness_judge.prompts"
COMPLETENESS_JUDGE_PROMPTS_FILE = "completeness_judge_prompt.jinja2"
COMPLETENESS_JUDGE_RESPONSE_SCHEMA_PATH = (
    EVALUATOR_DIR
    / "completeness_judge"
    / "prompts"
    / "completeness_judge_response_schema.json"
)

# Response classification evaluation results directory
RESPONSE_CLASSIFICATION_EVALUATION_RESULTS_DIR = (
    EVALUATOR_DIR / "response_classification_results"
)
# Default output filename
DEFAULT_RESPONSE_CLASSIFICATION_EVALUATION_TEXT_OUTPUT_FILENAME = "run_results.txt"
# Default YAML output filename
RESPONSE_CLASSIFICATION_EVALUATION_YAML_OUTPUT_FILENAME = "run_results.yaml"

# Copilot response evaluation results directory
COPILOT_RESPONSE_EVALUATION_RESULTS_DIR = EVALUATOR_DIR / "response_evaluation_results"
# Default output filename
DEFAULT_COPILOT_RESPONSE_EVALUATION_TEXT_OUTPUT_FILENAME = "run_results.txt"
# Default YAML output filename
COPILOT_RESPONSE_EVALUATION_YAML_OUTPUT_FILENAME = "run_results.yaml"
