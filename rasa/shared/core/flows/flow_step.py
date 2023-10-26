from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Generator,
    Optional,
    Set,
    Text,
)
import structlog

if TYPE_CHECKING:
    from rasa.shared.core.flows.flow_step_links import FlowStepLinks

structlogger = structlog.get_logger()


def step_from_json(flow_step_config: Dict[Text, Any]) -> FlowStep:
    """Used to read flow steps from parsed YAML.

    Args:
        flow_step_config: The parsed YAML as a dictionary.

    Returns:
        The parsed flow step.
    """
    from rasa.shared.core.flows.steps import (
        ActionFlowStep,
        UserMessageStep,
        CollectInformationFlowStep,
        LinkFlowStep,
        SetSlotsFlowStep,
        GenerateResponseFlowStep,
        BranchFlowStep,
    )

    if "action" in flow_step_config:
        return ActionFlowStep.from_json(flow_step_config)
    if "intent" in flow_step_config:
        return UserMessageStep.from_json(flow_step_config)
    if "collect" in flow_step_config:
        return CollectInformationFlowStep.from_json(flow_step_config)
    if "link" in flow_step_config:
        return LinkFlowStep.from_json(flow_step_config)
    if "set_slots" in flow_step_config:
        return SetSlotsFlowStep.from_json(flow_step_config)
    if "generation_prompt" in flow_step_config:
        return GenerateResponseFlowStep.from_json(flow_step_config)
    else:
        return BranchFlowStep.from_json(flow_step_config)


@dataclass
class FlowStep:
    """Represents the configuration of a flow step."""

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

    @classmethod
    def _from_json(cls, flow_step_config: Dict[Text, Any]) -> FlowStep:
        """Used to read flow steps from parsed YAML.

        Args:
            flow_step_config: The parsed YAML as a dictionary.

        Returns:
            The parsed flow step.
        """
        from rasa.shared.core.flows.flow_step_links import FlowStepLinks

        return FlowStep(
            # the idx is set later once the flow is created that contains
            # this step
            idx=-1,
            custom_id=flow_step_config.get("id"),
            description=flow_step_config.get("description"),
            metadata=flow_step_config.get("metadata", {}),
            next=FlowStepLinks.from_json(flow_step_config.get("next", [])),
        )

    def as_json(self) -> Dict[Text, Any]:
        """Returns the flow step as a dictionary.

        Returns:
            The flow step as a dictionary.
        """
        dump = {"next": self.next.as_json(), "id": self.id}

        if self.description:
            dump["description"] = self.description
        if self.metadata:
            dump["metadata"] = self.metadata
        return dump

    def steps_in_tree(self) -> Generator[FlowStep, None, None]:
        """Returns the steps in the tree of the flow step."""
        yield self
        yield from self.next.steps_in_tree()

    @property
    def id(self) -> Text:
        """Returns the id of the flow step."""
        return self.custom_id or self.default_id()

    def default_id(self) -> str:
        """Returns the default id of the flow step."""
        return f"{self.idx}_{self.default_id_postfix()}"

    def default_id_postfix(self) -> str:
        """Returns the default id postfix of the flow step."""
        raise NotImplementedError()

    @property
    def utterances(self) -> Set[str]:
        """Return all the utterances used in this step"""
        return set()
