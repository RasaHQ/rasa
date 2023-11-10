from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX


FLOW_PATTERN_CHITCHAT = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "chitchat"


@dataclass
class ChitchatPatternFlowStackFrame(PatternFlowStackFrame):
    """A flow stack frame that gets added to respond to Chitchat."""

    flow_id: str = FLOW_PATTERN_CHITCHAT
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CHITCHAT

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ChitchatPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return ChitchatPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
