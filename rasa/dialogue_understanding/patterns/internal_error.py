from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Text

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    RASA_PATTERN_INTERNAL_ERROR_DEFAULT,
)

FLOW_PATTERN_INTERNAL_ERROR_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "internal_error"


@dataclass
class InternalErrorPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an internal error occurs."""

    flow_id: str = FLOW_PATTERN_INTERNAL_ERROR_ID
    """The ID of the flow."""

    error_type: Optional[Text] = RASA_PATTERN_INTERNAL_ERROR_DEFAULT
    """Error type used in switch-case of the error pattern flow."""

    info: Dict[Text, Any] = field(default_factory=dict)
    """Additional info to be provided to the user"""

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
            info=data.get("info", {}),
        )
