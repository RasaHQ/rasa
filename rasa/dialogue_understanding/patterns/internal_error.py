from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Text

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    RASA_PATTERN_INTERNAL_ERROR_DEFAULT,
)

FLOW_PATTERN_INTERNAL_ERROR_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "internal_error"

INTERNAL_ERROR_SOURCE_AGENT = "agent"
"""Value for ``info["error_source"]`` when the failure came from an agent call."""

INTERNAL_ERROR_SOURCE_MCP_TOOL = "mcp_tool"
"""Value for ``info["error_source"]`` when the failure came from an MCP tool call."""

_INFO_KEYS_EXPOSED_IN_PATTERN_CONTEXT = (
    "error_source",
    "agent_name",
    "agent_type",
    "tool_name",
    "mcp_server",
    "error_message",
)


@dataclass
class InternalErrorPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an internal error occurs."""

    flow_id: str = FLOW_PATTERN_INTERNAL_ERROR_ID
    """The ID of the flow."""

    error_type: Optional[Text] = RASA_PATTERN_INTERNAL_ERROR_DEFAULT
    """Error type used in switch-case of the error pattern flow."""

    info: Dict[Text, Any] = field(default_factory=dict)
    """Structured failure context for ``pattern_internal_error`` (predicates, NLG).

    Contents vary by call site: many paths push an empty dict. Agent and MCP tool
    failures typically set ``error_source`` to ``INTERNAL_ERROR_SOURCE_AGENT`` or
    ``INTERNAL_ERROR_SOURCE_MCP_TOOL`` and add relevant metadata (for example
    ``agent_name``, ``tool_name``, ``mcp_server``, ``error_message``, ``flow_id``,
    ``step_id``). Other producers may use different keys altogether (for example
    ``max_characters`` when user input exceeds the configured limit).
    """

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_INTERNAL_ERROR_ID

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> InternalErrorPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return InternalErrorPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            error_type=data.get("error_type", RASA_PATTERN_INTERNAL_ERROR_DEFAULT),
            info=dict(data.get("info") or {}),
        )
