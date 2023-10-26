from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Text, Any

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class BranchFlowStep(FlowStep):
    """Represents the configuration of a branch flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> BranchFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return BranchFlowStep(**base.__dict__)

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        return dump

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "branch"
