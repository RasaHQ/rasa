from __future__ import annotations

from dataclasses import dataclass
from typing import Text

from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.constants import END_STEP, UNSET_FLOW_STEP_ID
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class EndFlowStep(InternalFlowStep):
    """A dynamically added flow step that marks the end of a flow."""

    def __init__(self, flow_id: Text) -> None:
        """Initializes an end flow step."""
        super().__init__(
            idx=UNSET_FLOW_STEP_ID,
            custom_id=END_STEP,
            description=None,
            metadata={},
            next=FlowStepLinks(links=[]),
            flow_id=flow_id,
        )
