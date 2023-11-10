from __future__ import annotations

from dataclasses import dataclass
from typing import Text

from rasa.shared.core.flows.flow_step_links import (
    FlowStepLinks,
    StaticFlowStepLink,
)
from rasa.shared.core.flows.steps.constants import START_STEP, UNSET_FLOW_STEP_ID
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class StartFlowStep(InternalFlowStep):
    """A dynamically added flow step that represents the beginning of a flow."""

    def __init__(self, start_step_id: Text) -> None:
        """Initializes a start flow step.

        Args:
            start_step_id: The step id of the first step of the flow
        """
        super().__init__(
            idx=UNSET_FLOW_STEP_ID,
            custom_id=START_STEP,
            description=None,
            metadata={},
            next=FlowStepLinks(links=[StaticFlowStepLink(start_step_id)]),
        )
