from __future__ import annotations

from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from dataclasses import dataclass
from typing import Any, Dict
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame


FLOW_PATTERN_INTERNAL_ERROR_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "internal_error"


@dataclass
class InternalErrorPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an internal error occurs."""

    flow_id: str = FLOW_PATTERN_INTERNAL_ERROR_ID
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "pattern_internal_error"

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
        )
