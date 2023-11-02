from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Text

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class SetSlotsFlowStep(FlowStep):
    """A flow step that sets one or multiple slots."""

    slots: List[Dict[str, Any]]
    """Slots and their values to set in the flow step."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> SetSlotsFlowStep:
        """Create a SetSlotsFlowStep from serialized data

        Args:
            data: data for a SetSlotsFlowStep in a serialized format

        Returns:
            a SetSlotsFlowStep object
        """
        base = super().from_json(data)
        slots = [
            {"key": k, "value": v}
            for slot_sets in data["set_slots"]
            for k, v in slot_sets.items()
        ]
        return SetSlotsFlowStep(
            slots=slots,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the SetSlotsFlowStep object

        Returns:
            the SetSlotsFlowStep object as serialized data
        """
        data = super().as_json()
        data["set_slots"] = [{slot["key"]: slot["value"]} for slot in self.slots]
        return data

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "set_slots"
