from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Text

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class SetSlotsFlowStep(FlowStep):
    """Represents the configuration of a set_slots flow step."""

    slots: List[Dict[str, Any]]
    """Slots to set of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> SetSlotsFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        slots = [
            {"key": k, "value": v}
            for slot in flow_step_config.get("set_slots", [])
            for k, v in slot.items()
        ]
        return SetSlotsFlowStep(
            slots=slots,
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["set_slots"] = [{slot["key"]: slot["value"]} for slot in self.slots]
        return dump

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "set_slots"
