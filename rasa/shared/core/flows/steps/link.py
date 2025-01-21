from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Text

from rasa.shared.core.flows.flow_step import FlowStep


@dataclass
class LinkFlowStep(FlowStep):
    """A flow step at the end of a flow that links to and starts another flow."""

    link: Text
    """The id of the flow that should be started subsequently."""

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

    def as_json(self) -> Dict[Text, Any]:
        """Serialize the LinkFlowStep object

        Returns:
            the LinkFlowStep object as serialized data.
        """
        data = super().as_json()
        data["link"] = self.link
        return data

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return f"link_{self.link}"
