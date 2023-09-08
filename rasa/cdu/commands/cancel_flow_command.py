from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.cdu.commands import Command
from rasa.cdu.patterns.cancel import CancelPatternFlowStackFrame
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames.flow_frame import BaseFlowStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.cdu.stack.utils import top_user_flow_frame

structlogger = structlog.get_logger()


@dataclass
class CancelFlowCommand(Command):
    """A command to cancel the current flow."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "cancel flow"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CancelFlowCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return CancelFlowCommand()

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        """Runs the command on the tracker.

        Args:
            tracker: The tracker to run the command on.
            all_flows: All flows in the assistant.
            original_tracker: The tracker before any command was executed.

        Returns:
            The events to apply to the tracker.
        """
        original_dialogue_stack = DialogueStack.from_tracker(original_tracker)
        original_stack_dump = original_dialogue_stack.as_dict()

        dialogue_stack = DialogueStack.from_tracker(tracker)
        current_user_frame = top_user_flow_frame(dialogue_stack)
        current_top_flow = (
            current_user_frame.flow(all_flows) if current_user_frame else None
        )
        if not current_top_flow:
            structlogger.debug(
                "command_executor.skip_cancel_flow.no_active_flow", command=self
            )
            return []

        canceled_frames = []
        original_frames = DialogueStack.from_dict(original_stack_dump).frames
        # we need to go through the original stack dump in reverse order
        # to find the frames that were canceled. we cancel everthing from
        # the top of the stack until we hit the user flow that was canceled.
        # this will also cancel any patterns put ontop of that user flow,
        # e.g. corrections.
        for frame in reversed(original_frames):
            canceled_frames.append(frame.frame_id)
            if (
                current_user_frame
                and isinstance(frame, BaseFlowStackFrame)
                and frame.flow_id == current_user_frame.flow_id
            ):
                break

        dialogue_stack.push(
            CancelPatternFlowStackFrame(
                canceled_name=current_user_frame.flow(all_flows).readable_name()
                if current_user_frame
                else None,
                canceled_frames=canceled_frames,
            )
        )
        return [SlotSet(DIALOGUE_STACK_SLOT, dialogue_stack.as_dict())]
