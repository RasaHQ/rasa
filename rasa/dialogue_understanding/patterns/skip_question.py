from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX

FLOW_PATTERN_SKIP_QUESTION = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "skip_question"


@dataclass
class SkipQuestionPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if user interrupts the current flow by
    trying to bypass the collect information step.
    """

    flow_id: str = FLOW_PATTERN_SKIP_QUESTION
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_SKIP_QUESTION

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SkipQuestionPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return SkipQuestionPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
