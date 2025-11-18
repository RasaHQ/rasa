import importlib.resources
from typing import Dict

import structlog
import yaml  # type: ignore[import-untyped]

from rasa.builder.copilot.constants import (
    COPILOT_HANDLER_RESPONSES_FILE,
    COPILOT_MESSAGE_TEMPLATES_DIR,
    COPILOT_TEMPLATE_PROMPTS_FILE,
    COPILOT_WELCOME_MESSAGES_FILE,
    RASA_INTERNAL_MESSAGES_TEMPLATES_FILE,
)
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()


def load_copilot_internal_message_templates() -> Dict[str, str]:
    """Load internal message templates from the YAML configuration file.

    Returns:
        Dictionary mapping template names to template text.
    """
    try:
        config = yaml.safe_load(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_MESSAGE_TEMPLATES_DIR}",
                RASA_INTERNAL_MESSAGES_TEMPLATES_FILE,
            )
        )
        return config.get("templates", {})
    except Exception as e:
        structlogger.error(
            "copilot_templated_message_provider.failed_to_load_templates",
            error=e,
        )
        return dict()


def load_copilot_handler_default_responses() -> Dict[str, str]:
    """Load handler responses from the YAML configuration file.

    Returns:
        Dictionary mapping response names to response text.
    """
    try:
        config = yaml.safe_load(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_MESSAGE_TEMPLATES_DIR}",
                COPILOT_HANDLER_RESPONSES_FILE,
            )
        )
        return config.get("responses", {})
    except Exception as e:
        structlogger.error(
            "copilot_response_handler.failed_to_load_responses",
            error=e,
        )
        return dict()


def load_copilot_welcome_messages() -> Dict[str, str]:
    """Load welcome message templates from the YAML configuration file.

    Returns:
        Dictionary mapping template names to welcome message text.
    """
    try:
        config = yaml.safe_load(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_MESSAGE_TEMPLATES_DIR}",
                COPILOT_WELCOME_MESSAGES_FILE,
            )
        )
        return config.get("welcome_messages", {})
    except Exception as e:
        structlogger.error(
            "copilot_templated_message_provider.failed_to_load_welcome_messages",
            error=e,
        )
        return dict()


def load_copilot_template_prompts() -> Dict[str, str]:
    """Load template prompt messages from the YAML configuration file.

    Returns:
        Dictionary mapping template names to template prompt text.
    """
    try:
        config = yaml.safe_load(
            importlib.resources.read_text(
                f"{PACKAGE_NAME}.{COPILOT_MESSAGE_TEMPLATES_DIR}",
                COPILOT_TEMPLATE_PROMPTS_FILE,
            )
        )
        return config.get("template_prompts", {})
    except Exception as e:
        structlogger.error(
            "copilot_templated_message_provider.failed_to_load_template_prompts",
            error=e,
        )
        return dict()
