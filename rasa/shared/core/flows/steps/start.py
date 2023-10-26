from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Text, List

from rasa.shared.core.flows.flow_step_links import (
    FlowStepLinks,
    FlowStepLink,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class StartFlowStep(InternalFlowStep):
    """A dynamically added flow step that represents the beginning of a flow."""

    def __init__(self, start_step_id: Optional[Text]) -> None:
        """Initializes a start flow step.

        Args:
            start_step_id: The step id of the first step of the flow
        """
        if start_step_id is not None:
            links: List[FlowStepLink] = [StaticFlowStepLink(start_step_id)]
        else:
            links = []

        super().__init__(
            idx=0,
            custom_id=START_STEP,
            description=None,
            metadata={},
            next=FlowStepLinks(links=links),
        )
