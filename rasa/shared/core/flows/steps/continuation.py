from __future__ import annotations

from dataclasses import dataclass

from rasa.shared.core.flows.flow_step_links import FlowLinks, StaticFlowLink
from rasa.shared.core.flows.steps.constants import CONTINUE_STEP_PREFIX
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class ContinueFlowStep(InternalFlowStep):
    """Represents the configuration of a continue-step flow step."""

    def __init__(self, next: str) -> None:
        """Initializes a continue-step flow step."""
        super().__init__(
            idx=0,
            custom_id=CONTINUE_STEP_PREFIX + next,
            description=None,
            metadata={},
            # The continue step links to the step that should be continued.
            # The flow policy in a sense only "runs" the logic of a step
            # when it transitions to that step, once it is there it will use
            # the next link to transition to the next step. This means that
            # if we want to "re-run" a step, we need to link to it again.
            # This is why the continue step links to the step that should be
            # continued.
            next=FlowLinks(links=[StaticFlowLink(next)]),
        )

    @staticmethod
    def continue_step_for_id(step_id: str) -> str:
        return CONTINUE_STEP_PREFIX + step_id
