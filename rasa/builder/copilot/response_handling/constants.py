"""Constants for copilot response handler.

This module contains shared constants and mappings used by both the legacy and
agent SDK response handlers.
"""

from typing import Dict, Set, Tuple

from rasa.builder.copilot.copilot_templated_message_provider import (
    copilot_handler_default_responses,
)
from rasa.builder.copilot.models import ResponseCategory

# Controlled prediction markers
ROLEPLAY_PREDICTION = "[ROLEPLAY_REQUEST_DETECTED]"
OUT_OF_SCOPE_PREDICTION = "[OUT_OF_SCOPE_REQUEST_DETECTED]"
ERROR_FALLBACK_PREDICTION = "[ERROR_FALLBACK]"
KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION = "[NO_KNOWLEDGE_BASE_ACCESS]"
RASA_INTRODUCTION_PREDICTION = "[RASA_INTRODUCTION_DETECTION]"
COPILOT_INTRODUCTION_PREDICTION = "[COPILOT_INTRODUCTION_DETECTION]"

# Response template keys (used to fetch responses from YAML)
GREETING_FALLBACK_RESPONSE_KEY = "greeting_fallback_response"
GOODBYE_FALLBACK_RESPONSE_KEY = "goodbye_fallback_response"
ROLEPLAY_RESPONSE_KEY = "roleplay_response"
OUT_OF_SCOPE_RESPONSE_KEY = "out_of_scope_response"
UNCLEAR_INPUT_RESPONSE_KEY = "unclear_input_response"
ERROR_FALLBACK_RESPONSE_KEY = "error_fallback_response"
KNOWLEDGE_BASE_ACCESS_REQUESTED_RESPONSE_KEY = (
    "knowledge_base_access_requested_response"
)
RASA_INTRODUCTION_RESPONSE_KEY = "rasa_introduction_response"
COPILOT_INTRODUCTION_RESPONSE_KEY = "copilot_introduction_response"

# Load predefined for controlled predictions from YAML
_handler_responses = copilot_handler_default_responses()

# Prediction marker to response mapping
PREDICTION_RESPONSES: Dict[str, Tuple[str, ResponseCategory]] = {
    ROLEPLAY_PREDICTION: (
        _handler_responses.get(ROLEPLAY_RESPONSE_KEY, ""),
        ResponseCategory.ROLEPLAY_DETECTION,
    ),
    OUT_OF_SCOPE_PREDICTION: (
        _handler_responses.get(OUT_OF_SCOPE_RESPONSE_KEY, ""),
        ResponseCategory.OUT_OF_SCOPE_DETECTION,
    ),
    ERROR_FALLBACK_PREDICTION: (
        _handler_responses.get(ERROR_FALLBACK_RESPONSE_KEY, ""),
        ResponseCategory.ERROR_FALLBACK,
    ),
    KNOWLEDGE_BASE_ACCESS_REQUESTED_PREDICTION: (
        _handler_responses.get(KNOWLEDGE_BASE_ACCESS_REQUESTED_RESPONSE_KEY, ""),
        ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
    ),
}

# Other response constants
GUARDRAIL_POLICY_VIOLATION_RESPONSE = _handler_responses.get(
    "guardrail_policy_violation_response", ""
)
COPILOT_REDACTED_MESSAGE = _handler_responses.get("copilot_redacted_message", "")

# Guardrails blocked responses
GUARDRAIL_BLOCKED_USER_RESPONSE = _handler_responses.get(
    "guardrail_blocked_user_response", ""
)
GUARDRAIL_BLOCKED_PROJECT_RESPONSE = _handler_responses.get(
    "guardrail_blocked_project_response", ""
)

# Exception response
EXCEPTION_RESPONSE = _handler_responses.get("exception_response", "")

# Regex pattern to match inline citations in the generated markdown.
INLINE_CITATION_PATTERN = r"\[([^\]]+)\]\(([^)]+)\)"

# Response categories that contain extractable text content (controlled predictions)
CONTROLLED_PREDICTION_CATEGORIES: Set[ResponseCategory] = {
    ResponseCategory.ROLEPLAY_DETECTION,
    ResponseCategory.OUT_OF_SCOPE_DETECTION,
    ResponseCategory.UNCLEAR_INPUT_DETECTION,
    ResponseCategory.ERROR_FALLBACK,
    ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
    ResponseCategory.RASA_INTRODUCTION_DETECTION,
    ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
}

# Primary text categories for legacy handler
LEGACY_PRIMARY_TEXT_CATEGORIES: Set[ResponseCategory] = {
    ResponseCategory.COPILOT,
}

# Primary text categories for agent handler
AGENT_PRIMARY_TEXT_CATEGORIES: Set[ResponseCategory] = {
    ResponseCategory.COPILOT_TEXT_CONTENT_PART_DELTA,
}

# Common LLM response prefixes and suffixes before the actual content. These are removed
# from the content.
LLM_PREFIXES_TO_SUFFIX_REMOVE = {
    "```markdown": "```",
    "```": "```",
    '"""': '"""',
}
