from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX

FLOW_PATTERN_USER_SILENCE = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "user_silence"


@dataclass
class UserSilencePatternFlowStackFrame(PatternFlowStackFrame):
    """A flow stack frame that can get added when the silence timeout trips."""

    flow_id: str = FLOW_PATTERN_USER_SILENCE
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_USER_SILENCE

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> UserSilencePatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return UserSilencePatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
