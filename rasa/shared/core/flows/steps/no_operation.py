from __future__ import annotations
from dataclasses import dataclass

from typing import Dict, Text, Any

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class NoOperationFlowStep(FlowStep):
    """A step that doesn't do a thing.

    This is NOT a branching step (but it can branch - but in addition to that
    it also does nothing)."""

    noop: Any
    """The id of the flow that should be started subsequently."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> NoOperationFlowStep:
        """Create a NoOperationFlowStep from serialized data

        Args:
            data: data for a NoOperationFlowStep in a serialized format

        Returns:
            a NoOperationFlowStep object
        """
        base = super().from_json(data)
        return NoOperationFlowStep(
            noop=data["noop"],
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the NoOperationFlowStep object

        Returns:
            the NoOperationFlowStep object as serialized data.
        """
        data = super().as_json()
        data["noop"] = self.noop
        return data

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "noop"
