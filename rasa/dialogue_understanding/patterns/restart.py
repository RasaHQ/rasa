from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX

FLOW_PATTERN_RESTART = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "restart"


@dataclass
class RestartPatternFlowStackFrame(PatternFlowStackFrame):
    """A flow stack frame that can get added at the beginning of the conversation."""

    flow_id: str = FLOW_PATTERN_RESTART
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_RESTART

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> RestartPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return RestartPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
