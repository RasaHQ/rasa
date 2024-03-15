from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Set

from rasa.shared.constants import UTTER_ASK_PREFIX, ACTION_ASK_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class SlotRejection:
    """A pair of validation condition and an utterance for the case of failure."""

    if_: str
    """The condition that should be checked."""
    utter: str
    """The utterance that should be executed if the condition is met."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SlotRejection:
        """Create a SlotRejection object from serialized data.

        Args:
            data: data for a SlotRejection object in a serialized format

        Returns:
            A SlotRejection object
        """
        return SlotRejection(
            if_=data["if"],
            utter=data["utter"],
        )

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the SlotRejection object.

        Returns:
            the SlotRejection object as serialized data
        """
        return {
            "if": self.if_,
            "utter": self.utter,
        }


@dataclass
class CollectInformationFlowStep(FlowStep):
    """A flow step for asking the user for information to fill a specific slot."""

    collect: str
    """The collect information of the flow step."""
    utter: str
    """The utterance that the assistant uses to ask for the slot."""
    collect_action: str
    """The action that the assistant uses to ask for the slot."""
    rejections: List[SlotRejection]
    """how the slot value is validated using predicate evaluation."""
    ask_before_filling: bool = False
    """Whether to always ask the question even if the slot is already filled."""
    reset_after_flow_ends: bool = True
    """Whether to reset the slot value at the end of the flow."""

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep object from serialized data.

        Args:
            data: data for a CollectInformationFlowStep object in a serialized format

        Returns:
            A CollectInformationFlowStep object
        """
        base = super().from_json(data)
        return CollectInformationFlowStep(
            collect=data["collect"],
            utter=data.get("utter", f"{UTTER_ASK_PREFIX}{data['collect']}"),
            # as of now it is not possible to define a different name for the
            # action, always use the default name 'action_ask_<slot_name>'
            collect_action=f"{ACTION_ASK_PREFIX}{data['collect']}",
            ask_before_filling=data.get("ask_before_filling", False),
            reset_after_flow_ends=data.get("reset_after_flow_ends", True),
            rejections=[
                SlotRejection.from_dict(rejection)
                for rejection in data.get("rejections", [])
            ],
            **base.__dict__,
        )

    def as_json(self) -> Dict[str, Any]:
        """Serialize the CollectInformationFlowStep object.

        Returns:
            the CollectInformationFlowStep object as serialized data
        """
        data = super().as_json()
        data["collect"] = self.collect
        data["utter"] = self.utter
        data["ask_before_filling"] = self.ask_before_filling
        data["reset_after_flow_ends"] = self.reset_after_flow_ends
        data["rejections"] = [rejection.as_dict() for rejection in self.rejections]

        return data

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"collect_{self.collect}"

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step."""
        return {self.utter} | {r.utter for r in self.rejections}
