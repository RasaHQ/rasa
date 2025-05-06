from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Text

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class NoOperationFlowStep(FlowStep):
    """A step that doesn't do a thing.

    This is NOT a branching step (but it can branch - but in addition to that
    it also does nothing).
    """

    noop: Any
    """The id of the flow that should be started subsequently."""

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> NoOperationFlowStep:
        """Create a NoOperationFlowStep from serialized data

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a NoOperationFlowStep in a serialized format

        Returns:
            a NoOperationFlowStep object
        """
        base = super().from_json(flow_id, data)
        return NoOperationFlowStep(
            noop=data["noop"],
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:  # type: ignore[override]
        """Serialize the NoOperationFlowStep object

        Returns:
            the NoOperationFlowStep object as serialized data.
        """
        return super().as_json(step_properties={"noop": self.noop})

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "noop"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.noop == other.noop and super().__eq__(other)
        return False
