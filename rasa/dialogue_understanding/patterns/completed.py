from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX


FLOW_PATTERN_COMPLETED = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "completed"


@dataclass
class CompletedPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which gets added if all prior flows are completed."""

    flow_id: str = FLOW_PATTERN_COMPLETED
    """The ID of the flow."""
    previous_flow_name: str = ""
    """The name of the last flow that was completed."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_COMPLETED

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CompletedPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CompletedPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            previous_flow_name=data["previous_flow_name"],
        )
