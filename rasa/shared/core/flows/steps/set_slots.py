from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Text

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class SetSlotsFlowStep(FlowStep):
    """A flow step that sets one or multiple slots."""

    slots: List[Dict[str, Any]]
    """Slots and their values to set in the flow step."""

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> SetSlotsFlowStep:
        """Create a SetSlotsFlowStep from serialized data

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a SetSlotsFlowStep in a serialized format

        Returns:
            a SetSlotsFlowStep object
        """
        base = super().from_json(flow_id, data)
        slots = [
            {"key": k, "value": v}
            for slot_sets in data["set_slots"]
            for k, v in slot_sets.items()
        ]
        return SetSlotsFlowStep(
            slots=slots,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:  # type: ignore[override]
        """Serialize the SetSlotsFlowStep object

        Returns:
            the SetSlotsFlowStep object as serialized data
        """
        set_slots = [{slot["key"]: slot["value"]} for slot in self.slots]
        return super().as_json(step_properties={"set_slots": set_slots})

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "set_slots"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.slots == other.slots and super().__eq__(other)
        return False
