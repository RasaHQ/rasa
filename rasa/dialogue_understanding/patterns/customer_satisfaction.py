from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_PATTERN_CUSTOMER_SATISFACTION

FLOW_PATTERN_CUSTOMER_SATISFACTION = RASA_PATTERN_CUSTOMER_SATISFACTION


@dataclass
class CustomerSatisfactionPatternFlowStackFrame(PatternFlowStackFrame):
    """A flow stack frame for collecting customer satisfaction feedback."""

    flow_id: str = FLOW_PATTERN_CUSTOMER_SATISFACTION
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CUSTOMER_SATISFACTION

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CustomerSatisfactionPatternFlowStackFrame:
        """Creates a `CustomerSatisfactionPatternFlowStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the frame from.

        Returns:
            The created `CustomerSatisfactionPatternFlowStackFrame`.
        """
        return CustomerSatisfactionPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )
