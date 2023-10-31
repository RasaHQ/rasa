from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX


FLOW_PATTERN_CONTINUE_INTERRUPTED = (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX + "continue_interrupted"
)


@dataclass
class ContinueInterruptedPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an interruption is completed."""

    flow_id: str = FLOW_PATTERN_CONTINUE_INTERRUPTED
    """The ID of the flow."""
    previous_flow_name: str = ""
    """The name of the flow that was interrupted."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CONTINUE_INTERRUPTED

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ContinueInterruptedPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return ContinueInterruptedPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            previous_flow_name=data["previous_flow_name"],
        )
