from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set, Text

from rasa.shared.constants import ACTION_ASK_PREFIX, UTTER_ASK_PREFIX
from rasa.shared.core.flows.flow_step import FlowStep
from rasa.shared.core.slots import SlotRejection


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
    force_slot_filling: bool = False
    """Whether to keep only the SetSlot command for the collected slot."""

    @classmethod
    def from_json(
        cls, flow_id: Text, data: Dict[str, Any]
    ) -> CollectInformationFlowStep:
        """Create a CollectInformationFlowStep object from serialized data.

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a CollectInformationFlowStep object in a serialized format

        Returns:
            A CollectInformationFlowStep object
        """
        base = super().from_json(flow_id, data)
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
            force_slot_filling=data.get("force_slot_filling", False),
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
        data["force_slot_filling"] = self.force_slot_filling

        return data

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"collect_{self.collect}"

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step."""
        return {self.utter} | {r.utter for r in self.rejections}
