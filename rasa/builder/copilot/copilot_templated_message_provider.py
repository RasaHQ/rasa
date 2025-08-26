import importlib.resources
from typing import Dict

import structlog
import yaml  # type: ignore

from rasa.builder.copilot.constants import (
    COPILOT_HANDLER_RESPONSES_FILE,
    COPILOT_MESSAGE_TEMPLATES_DIR,
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
