from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Text

from rasa.shared.core.flows.flow_step import FlowStep, Optional

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow import Flow


@dataclass
class LinkFlowStep(FlowStep):
    """A flow step at the end of a flow that links to and starts another flow."""

    link: Text
    """The id of the flow that should be started subsequently."""
    linked_flow_reference: Optional["Flow"] = None
    """The flow that is linked to by this step."""

    def does_allow_for_next_step(self) -> bool:
        """Returns whether this step allows for following steps.

        Link steps need to be terminal steps, so can't have a next step.
        """
        return False

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> LinkFlowStep:
        """Create a LinkFlowStep from serialized data

        Args:
            flow_id: The id of the flow that contains the step.
            data: data for a LinkFlowStep in a serialized format

        Returns:
            a LinkFlowStep object
        """
        base = super().from_json(flow_id, data)
        return LinkFlowStep(
            link=data["link"],
            **base.__dict__,
        )

    def as_json(self) -> Dict[Text, Any]:  # type: ignore[override]
        """Serialize the LinkFlowStep object

        Returns:
            the LinkFlowStep object as serialized data.
        """
        return super().as_json(step_properties={"link": self.link})

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"link_{self.link}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self.link == other.link and super().__eq__(other)
        return False
