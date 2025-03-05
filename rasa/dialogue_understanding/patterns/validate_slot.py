from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
)
from rasa.shared.constants import (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX,
    REFILL_UTTER,
    REJECTIONS,
)
from rasa.shared.core.slots import SlotRejection

FLOW_PATTERN_VALIDATE_SLOT = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "validate_slot"


@dataclass
class ValidateSlotPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame to validate slots."""

    flow_id: str = FLOW_PATTERN_VALIDATE_SLOT
    """The ID of the flow."""
    validate: str = ""
    """The slot that should be validated."""
    refill_utter: str = ""
    """The utter action that should be executed to ask the user to refill the
    information."""
    refill_action: str = ""
    """The action that should be executed to ask the user to refill the
    information."""
    rejections: List[SlotRejection] = field(default_factory=list)
    """The predicate check that should be applied to the filled slot.
    If a predicate check fails, its `utter` action indicated under rejections
    will be executed.
    """

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_VALIDATE_SLOT

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ValidateSlotPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        rejections = [
            SlotRejection.from_dict(rejection) for rejection in data.get(REJECTIONS, [])
        ]

        return ValidateSlotPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            validate=data["validate"],
            refill_action=data["refill_action"],
            refill_utter=data[REFILL_UTTER],
            rejections=rejections,
        )
