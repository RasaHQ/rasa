from __future__ import annotations

from typing import Optional, Dict, Any, List

from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.core.constants import ACTION_CLEAN_STACK, DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import ContinueFlowStep, END_STEP
from rasa.shared.core.trackers import DialogueStateTracker


class ActionCleanStack(action.Action):
    """Action which cancels a flow from the stack."""

    def name(self) -> str:
        """Return the flow name."""
        return ACTION_CLEAN_STACK

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Clean the stack."""
        stack = DialogueStack.from_tracker(tracker)

        new_frames = []
        # Set all frames to their end step, filter out any non-BaseFlowStackFrames
        for frame in stack.frames:
            if isinstance(frame, BaseFlowStackFrame):
                frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                if isinstance(frame, UserFlowStackFrame):
                    # Making sure there are no "continue interrupts" triggered
                    frame.frame_type = FlowStackFrameType.REGULAR
                new_frames.append(frame)
        new_stack = DialogueStack.from_dict([frame.as_dict() for frame in new_frames])

        return [SlotSet(DIALOGUE_STACK_SLOT, new_stack.as_dict())]
