from __future__ import annotations

from dataclasses import dataclass
from typing import Text, Dict, Any, Set

from rasa.shared.constants import UTTER_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class ActionFlowStep(FlowStep):
    """A flow step that that defines an action to be executed."""

    action: Text
    """The action of the flow step."""

    @classmethod
    def from_json(cls, data: Dict[Text, Any]) -> ActionFlowStep:
        """Create an ActionFlowStep object from serialized data

        Args:
            data: data for an ActionFlowStep object in a serialized format

        Returns:
            An ActionFlowStep object
        """
        base = super().from_json(data)
        return ActionFlowStep(
            action=data["action"],
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the ActionFlowStep

        Returns:
            The ActionFlowStep object as serialized data.
        """
        data = super().as_json()
        data["action"] = self.action
        return data

    @property
    def default_id_postfix(self) -> str:
        return self.action

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return {self.action} if self.action.startswith(UTTER_PREFIX) else set()
