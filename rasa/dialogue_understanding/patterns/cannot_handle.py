from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Text

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    RASA_PATTERN_CANNOT_HANDLE_DEFAULT,
)

FLOW_PATTERN_CANNOT_HANDLE = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "cannot_handle"


@dataclass
class CannotHandlePatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added when that the
    bot can't handle the user's input."""

    flow_id: str = FLOW_PATTERN_CANNOT_HANDLE
    """The ID of the flow."""

    reason: Optional[Text] = RASA_PATTERN_CANNOT_HANDLE_DEFAULT
    """Reason for cannot handle used in switch-case of the
    cannot handle pattern flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CANNOT_HANDLE

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CannotHandlePatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CannotHandlePatternFlowStackFrame(
            frame_id=data["frame_id"], step_id=data["step_id"], reason=data["reason"]
        )
