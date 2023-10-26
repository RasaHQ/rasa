from __future__ import annotations

from dataclasses import dataclass
from typing import Text, Dict, Any, Set

from rasa.shared.constants import UTTER_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class ActionFlowStep(FlowStep):
    """Represents the configuration of an action flow step."""

    action: Text
    """The action of the flow step."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> ActionFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return ActionFlowStep(
            action=flow_step_config.get("action", ""),
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["action"] = self.action
        return dump

    def default_id_postfix(self) -> str:
        return self.action

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return {self.action} if self.action.startswith(UTTER_PREFIX) else set()
