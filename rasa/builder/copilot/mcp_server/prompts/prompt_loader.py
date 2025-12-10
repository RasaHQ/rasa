"""MCP prompts exposing existing copilot prompt templates."""

import importlib.resources
from typing import Dict

import structlog

from rasa.builder.copilot.constants import (
    COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE,
    COPILOT_PROMPTS_DIR,
    COPILOT_PROMPTS_FILE,
    COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE,
)
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()


# Cache for loaded prompt template sources
_prompt_sources: Dict[str, str] = {}


def _load_prompt_source(filename: str) -> str:
    """Load a prompt template source from the prompts directory.

    The loaded template is cached in memory to avoid repeated disk reads on
    subsequent calls.
    """
    if filename not in _prompt_sources:
        template_content = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{COPILOT_PROMPTS_DIR}",
            filename,
        )
        _prompt_sources[filename] = template_content
    return _prompt_sources[filename]


async def get_copilot_system_prompt() -> str:
    """Get the main copilot system prompt template.

    Returns:
        The rendered system prompt for the copilot.
    """
    try:
        # Return the raw template content (without rendering, as rendering
        # might require context variables)
        return _load_prompt_source(COPILOT_PROMPTS_FILE)
    except Exception as e:
        structlogger.error(
            "mcp_server.prompts.prompt_loader.get_copilot_system_prompt.error",
            event_info="MCP prompt failed to get copilot system prompt",
            error=str(e),
        )
        return f"Error loading system prompt: {e!s}"


async def get_last_user_message_context_prompt() -> str:
    """Get the template for adding context to the last user message.

    Returns:
        The template for contextualizing user messages.
    """
    try:
        return _load_prompt_source(COPILOT_LAST_USER_MESSAGE_CONTEXT_PROMPT_FILE)
    except Exception as e:
        structlogger.error(
            "mcp_server.prompts.prompt_loader.get_last_user_message_context_prompt.error",
            event_info="MCP prompt failed to get user message context prompt",
            error=str(e),
        )
        return f"Error loading user message context prompt: {e!s}"


async def get_training_error_handler_prompt() -> str:
    """Get the template for training error analysis.

    Returns:
        The template for analyzing training errors.
    """
    try:
        return _load_prompt_source(COPILOT_TRAINING_ERROR_HANDLER_PROMPT_FILE)
    except Exception as e:
        structlogger.error(
            "mcp_server.prompts.prompt_loader.get_training_error_handler_prompt.error",
            event_info="MCP prompt failed to get training error handler prompt",
            error=str(e),
        )
        return f"Error loading training error handler prompt: {e!s}"
