from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Text, List

from rasa.shared.core.flows.flow_step_links import FlowLinks, FlowLink, StaticFlowLink
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class StartFlowStep(InternalFlowStep):
    """Represents the configuration of a start flow step."""

    def __init__(self, start_step_id: Optional[Text]) -> None:
        """Initializes a start flow step.

        Args:
            start_step: The step to start the flow from.
        """
        if start_step_id is not None:
            links: List[FlowLink] = [StaticFlowLink(start_step_id)]
        else:
            links = []

        super().__init__(
            idx=0,
            custom_id=START_STEP,
            description=None,
            metadata={},
            next=FlowLinks(links=links),
        )
