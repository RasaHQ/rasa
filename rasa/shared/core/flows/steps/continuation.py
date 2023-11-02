from __future__ import annotations

from dataclasses import dataclass

from rasa.shared.core.flows.flow_step_links import FlowStepLinks, StaticFlowStepLink
from rasa.shared.core.flows.steps.constants import (
    CONTINUE_STEP_PREFIX,
    UNSET_FLOW_STEP_ID,
)
from rasa.shared.core.flows.steps.internal import InternalFlowStep


@dataclass
class ContinueFlowStep(InternalFlowStep):
    """A flow step that is dynamically introduced to jump to other flow steps."""

    def __init__(self, target_step_id: str) -> None:
        """Initializes a continue-step flow step."""
        super().__init__(
            idx=UNSET_FLOW_STEP_ID,
            custom_id=CONTINUE_STEP_PREFIX + target_step_id,
            description=None,
            metadata={},
            # The continue step links to the step that should be continued.
            # The flow policy in a sense only "runs" the logic of a step
            # when it transitions to that step, once it is there it will use
            # the next link to transition to the next step. This means that
            # if we want to "re-run" a step, we need to link to it again.
            # This is why the continue step links to the step that should be
            # continued.
            next=FlowStepLinks(links=[StaticFlowStepLink(target_step_id)]),
        )

    @staticmethod
    def continue_step_for_id(step_id: str) -> str:
        return CONTINUE_STEP_PREFIX + step_id
