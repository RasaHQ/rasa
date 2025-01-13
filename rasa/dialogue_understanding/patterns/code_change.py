from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import structlog

from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
)
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX

structlogger = structlog.get_logger()

FLOW_PATTERN_CODE_CHANGE_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "code_change"


@dataclass
class CodeChangeFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which cleans the stack after a bot update."""

    flow_id: str = FLOW_PATTERN_CODE_CHANGE_ID
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CODE_CHANGE_ID

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CodeChangeFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CodeChangeFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
