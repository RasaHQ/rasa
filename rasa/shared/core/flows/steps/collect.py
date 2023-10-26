from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Text, Any, List, Set

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class SlotRejection:
    """A slot rejection."""

    if_: str
    """The condition that should be checked."""
    utter: str
    """The utterance that should be executed if the condition is met."""

    @staticmethod
    def from_dict(rejection_config: Dict[Text, Any]) -> SlotRejection:
        """Used to read slot rejections from parsed YAML.

        Args:
            rejection_config: The parsed YAML as a dictionary.

        Returns:
            The parsed slot rejection.
        """
        return SlotRejection(
            if_=rejection_config["if"],
            utter=rejection_config["utter"],
        )

    def as_dict(self) -> Dict[Text, Any]:
        """Returns the slot rejection as a dictionary.

        Returns:
            The slot rejection as a dictionary.
        """
        return {
            "if": self.if_,
            "utter": self.utter,
        }


@dataclass
class CollectInformationFlowStep(FlowStep):
    """Represents the configuration of a collect information flow step."""

    collect: Text
    """The collect information of the flow step."""
    utter: Text
    """The utterance that the assistant uses to ask for the slot."""
    rejections: List[SlotRejection]
    """how the slot value is validated using predicate evaluation."""
    ask_before_filling: bool = False
    """Whether to always ask the question even if the slot is already filled."""
    reset_after_flow_ends: bool = True
    """Determines whether to reset the slot value at the end of the flow."""

    @classmethod
    def from_json(cls, flow_step_config: Dict[Text, Any]) -> CollectInformationFlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        base = super()._from_json(flow_step_config)
        return CollectInformationFlowStep(
            collect=flow_step_config["collect"],
            utter=flow_step_config.get(
                "utter", f"utter_ask_{flow_step_config['collect']}"
            ),
            ask_before_filling=flow_step_config.get("ask_before_filling", False),
            reset_after_flow_ends=flow_step_config.get("reset_after_flow_ends", True),
            rejections=[
                SlotRejection.from_dict(rejection)
                for rejection in flow_step_config.get("rejections", [])
            ],
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = super().as_json()
        dump["collect"] = self.collect
        dump["utter"] = self.utter
        dump["ask_before_filling"] = self.ask_before_filling
        dump["reset_after_flow_ends"] = self.reset_after_flow_ends
        dump["rejections"] = [rejection.as_dict() for rejection in self.rejections]

        return dump

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"collect_{self.collect}"

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return {self.utter} | {r.utter for r in self.rejections}
