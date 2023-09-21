from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
    BaseFlowStackFrame,
)
from rasa.core.actions import action
from rasa.core.channels.channel import OutputChannel
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import ACTION_CANCEL_FLOW, DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import END_STEP, ContinueFlowStep
from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()

FLOW_PATTERN_CANCEL_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "cancel_flow"


@dataclass
class CancelPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which cancels a flow.

    The frame contains the information about the stack frames that should
    be canceled."""

    flow_id: str = FLOW_PATTERN_CANCEL_ID
    """The ID of the flow."""
    canceled_name: str = ""
    """The name of the flow that should be canceled."""
    canceled_frames: List[str] = field(default_factory=list)
    """The stack frames that should be canceled. These can be multiple
    frames since the user frame that is getting canceled might have
    created patterns that should be canceled as well."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "pattern_cancel_flow"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CancelPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CancelPatternFlowStackFrame(
            data["frame_id"],
            step_id=data["step_id"],
            canceled_name=data["canceled_name"],
            canceled_frames=data["canceled_frames"],
        )


class ActionCancelFlow(action.Action):
    """Action which cancels a flow from the stack."""

    def __init__(self) -> None:
        """Creates a `ActionCancelFlow`."""
        super().__init__()

    def name(self) -> str:
        """Return the flow name."""
        return ACTION_CANCEL_FLOW

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Cancel the flow."""
        stack = DialogueStack.from_tracker(tracker)
        if not (top := stack.top()):
            structlogger.warning("action.cancel_flow.no_active_flow")
            return []

        if not isinstance(top, CancelPatternFlowStackFrame):
            structlogger.warning("action.cancel_flow.no_cancel_frame", top=top)
            return []

        for canceled_frame_id in top.canceled_frames:
            for frame in stack.frames:
                if frame.frame_id == canceled_frame_id and isinstance(
                    frame, BaseFlowStackFrame
                ):
                    # Setting the stack frame to the end step so it is properly
                    # wrapped up by the flow policy
                    frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                    break
            else:
                structlogger.warning(
                    "action.cancel_flow.frame_not_found",
                    stack=stack,
                    frame_id=canceled_frame_id,
                )

        return [SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())]
