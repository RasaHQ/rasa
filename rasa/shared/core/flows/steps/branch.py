from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Text, Any

from rasa.shared.core.flows.flow_step import FlowStep

# TODO: this is ambiguous, even steps that only have one static
#  follow up might become branch flow steps
#  validator also misuses it!!!
@dataclass
class BranchFlowStep(FlowStep):
    """An unspecific FlowStep that has a next attribute."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> BranchFlowStep:
        """Create a BranchFlowStep object from serialized data.

        Args:
            data: data for a BranchFlowStep object in a serialized format

        Returns:
            A BranchFlowStep object.
        """
        base = super()._from_json(data)
        return BranchFlowStep(**base.__dict__)

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the BranchFlowStep object

        Returns:
            the BranchFlowStep object as serialized data.
        """
        dump = super().as_json()
        return dump

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "branch"
