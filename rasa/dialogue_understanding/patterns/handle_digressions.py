from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import (
    KEY_ASK_CONFIRM_DIGRESSIONS,
    KEY_BLOCK_DIGRESSIONS,
)

FLOW_PATTERN_HANDLE_DIGRESSIONS = (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX + "handle_digressions"
)


@dataclass
class HandleDigressionsPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an interruption is completed."""

    flow_id: str = FLOW_PATTERN_HANDLE_DIGRESSIONS
    """The ID of the flow."""
    interrupting_flow_id: str = ""
    """The ID of the flow that interrupted the active flow."""
    interrupted_flow_id: str = ""
    """The name of the active flow that was interrupted."""
    interrupted_step_id: str = ""
    """The ID of the step that was interrupted."""
    ask_confirm_digressions: Set[str] = field(default_factory=set)
    """The set of interrupting flow names to confirm."""
    block_digressions: Set[str] = field(default_factory=set)
    """The set of interrupting flow names to block."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_HANDLE_DIGRESSIONS

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> HandleDigressionsPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return HandleDigressionsPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            interrupted_step_id=data["interrupted_step_id"],
            interrupted_flow_id=data["interrupted_flow_id"],
            interrupting_flow_id=data["interrupting_flow_id"],
            ask_confirm_digressions=set(data.get(KEY_ASK_CONFIRM_DIGRESSIONS, [])),
            # This attribute must be converted to a set to enable usage
            # of subset `contains` pypred operator in the default pattern
            # conditional branching
            block_digressions=set(data.get(KEY_BLOCK_DIGRESSIONS, [])),
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, HandleDigressionsPatternFlowStackFrame):
            return False
        return (
            self.flow_id == other.flow_id
            and self.interrupted_step_id == other.interrupted_step_id
            and self.interrupted_flow_id == other.interrupted_flow_id
            and self.interrupting_flow_id == other.interrupting_flow_id
            and self.ask_confirm_digressions == other.ask_confirm_digressions
            and self.block_digressions == other.block_digressions
        )

    def as_dict(self) -> Dict[str, Any]:
        """Returns the frame as a dictionary."""
        data = super().as_dict()
        # converting back to list to avoid serialization issues
        data[KEY_ASK_CONFIRM_DIGRESSIONS] = list(self.ask_confirm_digressions)
        data[KEY_BLOCK_DIGRESSIONS] = list(self.block_digressions)
        return data
