from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import (
    PatternFlowStackFrame,
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.core.actions import action
from rasa.core.channels.channel import OutputChannel
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT, ACTION_CLEAN_STACK
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import END_STEP, ContinueFlowStep
from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()

FLOW_PATTERN_CLEAN_STACK_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "clean_stack"


@dataclass
class CleanStackFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which cleans the stack after a bot update."""

    flow_id: str = FLOW_PATTERN_CLEAN_STACK_ID
    """The ID of the flow."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "pattern_clean_stack"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> CleanStackFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CleanStackFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
        )


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
        if not (top := stack.top()):
            structlogger.warning("action.clean_stack.no_active_flow")
            return []

        if not isinstance(top, CleanStackFlowStackFrame):
            structlogger.warning("action.clean_stack.no_cleaning_frame", top=top)
            return []

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
