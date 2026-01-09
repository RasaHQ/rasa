from typing import Type

# Always import both copilot implementations
from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.legacy_copilot import LegacyCopilot
from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
    AgentCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.base_copilot_response_handler import (
    BaseCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.legacy_copilot_response_handler import (
    LegacyCopilotResponseHandler,
)

# Runtime copilot mode - can be changed at runtime
# This will be set by the Sanic app context
_runtime_copilot_mode: str = "legacy"  # Default to legacy


def set_copilot_mode(mode: str) -> None:
    """Set the runtime copilot mode.

    Args:
        mode: Either "agent_sdk" or "legacy"
    """
    global _runtime_copilot_mode
    if mode not in ("agent_sdk", "legacy"):
        raise ValueError(
            f"Invalid copilot mode: {mode}. Must be 'agent_sdk' or 'legacy'"
        )
    _runtime_copilot_mode = mode


def get_copilot_mode() -> str:
    """Get the current runtime copilot mode.

    Returns:
        Current mode: either "agent_sdk" or "legacy"
    """
    return _runtime_copilot_mode


def get_copilot_class() -> Type[BaseCopilot]:
    """Get the appropriate copilot class based on runtime mode.

    Returns:
        AgentCopilot or LegacyCopilot class
    """
    if _runtime_copilot_mode == "agent_sdk":
        return AgentCopilot
    return LegacyCopilot


def get_copilot_response_handler_class() -> Type[BaseCopilotResponseHandler]:
    """Get the appropriate copilot response handler class based on runtime mode.

    Returns:
        AgentCopilotResponseHandler or LegacyCopilotResponseHandler class
    """
    if _runtime_copilot_mode == "agent_sdk":
        return AgentCopilotResponseHandler
    return LegacyCopilotResponseHandler


# Backward compatibility: Keep the old names but mark as deprecated
# These should be replaced with factory functions in calling code
Copilot: Type[BaseCopilot] = LegacyCopilot
CopilotResponseHandler: Type[BaseCopilotResponseHandler] = LegacyCopilotResponseHandler


__all__ = [
    "Copilot",
    "CopilotResponseHandler",
    "BaseCopilot",
    "BaseCopilotResponseHandler",
    "get_copilot_class",
    "get_copilot_response_handler_class",
    "set_copilot_mode",
    "get_copilot_mode",
    "AgentCopilot",
    "LegacyCopilot",
    "AgentCopilotResponseHandler",
    "LegacyCopilotResponseHandler",
]
