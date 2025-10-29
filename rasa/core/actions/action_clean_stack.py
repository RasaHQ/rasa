from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

import rasa.dialogue_understanding.stack.utils
from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.code_change import FLOW_PATTERN_CODE_CHANGE_ID
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.core.constants import ACTION_CLEAN_STACK
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.flows.steps.constants import END_STEP
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


class ActionCleanStack(Action):
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
        structlogger.debug("action_clean_stack.run")
        new_frames = []
        top_flow_frame = rasa.dialogue_understanding.stack.utils.top_flow_frame(
            tracker.stack, ignore_call_frames=False
        )
        top_user_flow_frame = (
            rasa.dialogue_understanding.stack.utils.top_user_flow_frame(
                tracker.stack, ignore_call_and_link_frames=False
            )
        )

        # Set all frames to their end step, filter out any non-BaseFlowStackFrames
        for frame in tracker.stack.frames:
            if isinstance(frame, BaseFlowStackFrame):
                structlogger.debug(
                    "action_clean_stack.terminating_frame",
                    frame_id=frame.frame_id,
                    flow_id=frame.flow_id,
                )
                frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                if isinstance(frame, UserFlowStackFrame):
                    # Making sure there are no "continue interrupts" triggered
                    frame.frame_type = FlowStackFrameType.REGULAR
                new_frames.append(frame)
        new_stack = DialogueStack.from_dict([frame.as_dict() for frame in new_frames])

        # Check if the action is being called from within a user flow
        if (
            top_flow_frame
            and top_flow_frame.flow_id != FLOW_PATTERN_CODE_CHANGE_ID
            and top_user_flow_frame
            and top_user_flow_frame.flow_id == top_flow_frame.flow_id
        ):
            # The action is being called from within a user flow on the stack.
            # If there are other frames on the stack, we need to make sure
            # the last executed frame is the end step of the current user flow so
            # that we can trigger pattern_completed for this user flow.
            new_stack.pop()
            structlogger.debug(
                "action_clean_stack.pushing_user_frame_at_the_bottom_of_stack",
                flow_id=top_user_flow_frame.flow_id,
            )
            new_stack.push(
                top_user_flow_frame,
                index=0,
            )

        return tracker.create_stack_updated_events(new_stack)
