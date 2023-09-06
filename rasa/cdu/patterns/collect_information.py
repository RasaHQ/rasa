from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from rasa.cdu.stack.dialogue_stack import DialogueStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.cdu.stack.frames import PatternFlowStackFrame

FLOW_PATTERN_COLLECT_INFORMATION = (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX + "ask_collect_information"
)


@dataclass
class CollectInformationPatternFlowStackFrame(PatternFlowStackFrame):
    flow_id: str = FLOW_PATTERN_COLLECT_INFORMATION
    collect_information: str = ""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "pattern_collect_information"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CollectInformationPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CollectInformationPatternFlowStackFrame(
            data["frame_id"],
            step_id=data["step_id"],
            collect_information=data["collect_information"],
        )

    def as_dict(self) -> Dict[str, Any]:
        super_dict = super().as_dict()
        super_dict.update(
            {
                "collect_information": self.collect_information,
            }
        )
        return super_dict

    def context_as_dict(
        self, underlying_frames: List[DialogueStackFrame]
    ) -> Dict[str, Any]:
        super_dict = super().context_as_dict(underlying_frames)

        if underlying_frames:
            underlying_context = underlying_frames[-1].context_as_dict(
                underlying_frames[:-1]
            )
        else:
            underlying_context = {}

        # the collect information frame is a special case, as it is not
        # a regular frame, but a frame that is used to collect information

        super_dict.update(underlying_context)
        super_dict.update(
            {
                "collect_information": self.collect_information,
            }
        )
        return super_dict
