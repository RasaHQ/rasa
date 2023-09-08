from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Set

import structlog
from rasa.cdu.commands import Command
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames.flow_frame import FlowStackFrameType, UserFlowStackFrame
from rasa.cdu.stack.utils import top_user_flow_frame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class StartFlowCommand(Command):
    """A command to start a flow."""

    flow: str

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "start flow"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StartFlowCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return StartFlowCommand(flow=data["flow"])

    @staticmethod
    def _all_user_flows_on_the_stack(dialogue_stack: DialogueStack) -> Set[str]:
        """Get all user flows that are currently on the stack.

        Args:
            dialogue_stack: The dialogue stack.

        Returns:
            All user flows that are currently on the stack."""
        return {
            f.flow_id
            for f in dialogue_stack.frames
            if isinstance(f, UserFlowStackFrame)
        }

    @staticmethod
    def _all_startable_flows(all_flows: FlowsList) -> List[str]:
        """Get all flows that can be started.

        Args:
            all_flows: All flows.

        Returns:
            All flows that can be started."""
        return [f.id for f in all_flows.underlying_flows if not f.is_handling_pattern()]

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
        dialogue_stack = DialogueStack.from_tracker(tracker)
        original_dialogue_stack = DialogueStack.from_tracker(original_tracker)

        if self.flow in self._all_user_flows_on_the_stack(dialogue_stack):
            structlogger.debug(
                "command_executor.skip_command.already_started_flow", command=self
            )
            return []
        elif self.flow not in self._all_startable_flows(all_flows):
            structlogger.debug(
                "command_executor.skip_command.start_invalid_flow_id", command=self
            )
            return []

        original_user_frame = top_user_flow_frame(original_dialogue_stack)
        original_top_flow = (
            original_user_frame.flow(all_flows) if original_user_frame else None
        )
        frame_type = (
            FlowStackFrameType.INTERRUPT
            if original_top_flow
            else FlowStackFrameType.REGULAR
        )
        structlogger.debug("command_executor.start_flow", command=self)
        dialogue_stack.push(
            UserFlowStackFrame(flow_id=self.flow, frame_type=frame_type)
        )
        return [SlotSet(DIALOGUE_STACK_SLOT, dialogue_stack.as_dict())]
