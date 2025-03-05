from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rasa.dialogue_understanding.stack.frames import (
    DialogueStackFrame,
    PatternFlowStackFrame,
)
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.slots import SlotRejection

FLOW_PATTERN_COLLECT_INFORMATION = (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX + "collect_information"
)


@dataclass
class CollectInformationPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which collects information from the user."""

    flow_id: str = FLOW_PATTERN_COLLECT_INFORMATION
    """The ID of the flow."""
    collect: str = ""
    """The information that should be collected from the user.
    this corresponds to the slot that will be filled."""
    utter: str = ""
    """The utter action that should be executed to ask the user for the
    information."""
    collect_action: str = ""
    """The action that should be executed to ask the user for the
    information."""
    rejections: Optional[List[SlotRejection]] = None
    """The predicate check that should be applied to the collected information.
    If a predicate check fails, its `utter` action indicated under rejections
    will be executed.
    """

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
        rejections = data.get("rejections")
        if rejections is not None:
            rejections = [
                SlotRejection.from_dict(rejection) for rejection in rejections
            ]

        return CollectInformationPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            collect=data["collect"],
            collect_action=data["collect_action"],
            utter=data["utter"],
            rejections=rejections,
        )

    def context_as_dict(
        self, underlying_frames: List[DialogueStackFrame]
    ) -> Dict[str, Any]:
        """Returns the context of the frame as a dictionary.

        The collect information frame needs a special implementation as
        it includes the context of the underlying frame in its context.

        This corresponds to the user expectation when e.g. using templates
        in a collect information node.
        """
        context = super().context_as_dict(underlying_frames)

        if underlying_frames:
            underlying_context = underlying_frames[-1].context_as_dict(
                underlying_frames[:-1]
            )
        else:
            underlying_context = {}

        # the collect information frame is a special case, as it is not
        # a regular frame, but a frame that is used to collect information

        context.update(underlying_context)
        return context
