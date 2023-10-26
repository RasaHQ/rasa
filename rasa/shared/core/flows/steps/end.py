from __future__ import annotations

from dataclasses import dataclass

from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.constants import END_STEP
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class EndFlowStep(InternalFlowStep):
    """A dynamically added flow step that marks the end of a flow."""

    def __init__(self) -> None:
        """Initializes an end flow step."""
        super().__init__(
            idx=0,
            custom_id=END_STEP,
            description=None,
            metadata={},
            next=FlowStepLinks(links=[]),
        )
