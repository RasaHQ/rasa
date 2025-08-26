from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Optional,
    Set,
    Text,
)

import structlog

from rasa.shared.exceptions import RasaException

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow_step_links import FlowStepLinks

structlogger = structlog.get_logger()


def step_from_json(flow_id: Text, data: Dict[Text, Any]) -> FlowStep:
    """Create a specific FlowStep from serialized data.

    Args:
        flow_id: The id of the flow that contains the step.
        data: data for a specific FlowStep object in a serialized data format.

    Returns:
        An instance of a specific FlowStep class.
    """
    from rasa.shared.core.flows.steps import (
        ActionFlowStep,
        CallFlowStep,
        CollectInformationFlowStep,
        LinkFlowStep,
        NoOperationFlowStep,
        SetSlotsFlowStep,
    )

    if "action" in data:
        return ActionFlowStep.from_json(flow_id, data)
    if "collect" in data:
        return CollectInformationFlowStep.from_json(flow_id, data)
    if "link" in data:
        return LinkFlowStep.from_json(flow_id, data)
    if "call" in data:
        return CallFlowStep.from_json(flow_id, data)
    if "set_slots" in data:
        return SetSlotsFlowStep.from_json(flow_id, data)
    if "noop" in data:
        return NoOperationFlowStep.from_json(flow_id, data)

    required_properties = ["action", "collect", "link", "call", "set_slots", "noop"]
    raise RasaException(
        f"Failed to parse step from json. Unknown type for {data}. "
        f"At least one of the following properties is required: "
        f"{', '.join(required_properties)}"
    )


@dataclass
class FlowStep:
    """A single step in a flow."""

    custom_id: Optional[Text]
    """The id of the flow step."""
    idx: int
    """The index of the step in the flow."""
    description: Optional[Text]
    """The description of the flow step."""
    metadata: Dict[Text, Any]
    """Additional, unstructured information about this flow step."""
    next: FlowStepLinks
    """The next steps of the flow step."""
    flow_id: Text
    """The id of the flow that contains the step."""

    @classmethod
    def from_json(cls, flow_id: Text, data: Dict[Text, Any]) -> FlowStep:
        """Create a FlowStep object from data in a serialized format.

        Args:
            flow_id: The id of the flow that contains the step.
            data: The data for a FlowStep object in a serialized format.

        Returns:
            The FlowStep object.
        """
        from rasa.shared.core.flows.flow_step_links import FlowStepLinks
        from rasa.shared.core.flows.steps.constants import UNSET_FLOW_STEP_ID

        return FlowStep(
            # the idx is set once the flow, which contains this step, is created
            idx=UNSET_FLOW_STEP_ID,
            custom_id=data.get("id"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            next=FlowStepLinks.from_json(flow_id, data.get("next", [])),
            flow_id=flow_id,
        )

    def does_allow_for_next_step(self) -> bool:
        """Most steps allow linking to the next step. But some don't."""
        return True

    def as_json(
        self, step_properties: Optional[Dict[Text, Any]] = None
    ) -> Dict[Text, Any]:
        """Serialize the FlowStep object.

        Returns:
            The FlowStep as serialized data.
        """
        data: Dict[Text, Any] = {"id": self.id}

        if step_properties:
            data.update(step_properties)

        if self.description:
            data["description"] = self.description
        if dumped_next := self.next.as_json():
            data["next"] = dumped_next
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def steps_in_tree(
        self, should_resolve_calls: bool = True
    ) -> Generator[FlowStep, None, None]:
        """Recursively generates the steps in the tree."""
        yield self
        yield from self.next.steps_in_tree(should_resolve_calls)

    @property
    def id(self) -> Text:
        """Returns the id of the flow step."""
        return self.custom_id or self.default_id

    @property
    def default_id(self) -> str:
        """Returns the default id of the flow step."""
        return f"{self.flow_id}_{self.idx}_{self.default_id_postfix}"

    @property
    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        return "step"

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step."""
        return set()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowStep):
            return False

        return (
            self.idx == other.idx
            and self.description == other.description
            and self.next == other.next
            and self.flow_id == other.flow_id
        )


@dataclass
class FlowStepWithFlowReference:
    step: FlowStep
    """The step."""
    flow_id: str
    """The id of the flow that contains the step."""
