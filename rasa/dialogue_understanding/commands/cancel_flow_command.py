from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.core.events import Event, FlowCancelled
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.dialogue_understanding.stack.utils import top_user_flow_frame

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

    @staticmethod
    def select_canceled_frames(stack: DialogueStack) -> List[str]:
        """Selects the frames that were canceled.

        Args:
            dialogue_stack: The dialogue stack.
            current_flow: The current flow.

        Returns:
            The frames that were canceled."""
        canceled_frames = []
        # we need to go through the original stack dump in reverse order
        # to find the frames that were canceled. we cancel everything from
        # the top of the stack until we hit the user flow that was canceled.
        # this will also cancel any patterns put on top of that user flow,
        # e.g. corrections.
        for frame in reversed(stack.frames):
            canceled_frames.append(frame.frame_id)
            if (
                isinstance(frame, UserFlowStackFrame)
                and frame.frame_type != FlowStackFrameType.CALL
            ):
                return canceled_frames
        else:
            # we should never get here as we should always find the user flow
            # that was canceled.
            raise ValueError(
                f"Could not find a user flow frame to cancel. "
                f"Current stack: {stack}."
            )

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

        stack = tracker.stack
        original_stack = original_tracker.stack

        applied_events: List[Event] = []

        user_frame = top_user_flow_frame(original_stack)
        current_flow = user_frame.flow(all_flows) if user_frame else None

        if not current_flow:
            structlogger.debug(
                "command_executor.skip_cancel_flow.no_active_flow", command=self
            )
            return []

        # we pass in the original dialogue stack (before any of the currently
        # predicted commands were applied) to make sure we don't cancel any
        # frames that were added by the currently predicted commands.
        canceled_frames = self.select_canceled_frames(original_stack)

        stack.push(
            CancelPatternFlowStackFrame(
                canceled_name=current_flow.readable_name(),
                canceled_frames=canceled_frames,
            )
        )

        if user_frame:
            applied_events.append(FlowCancelled(user_frame.flow_id, user_frame.step_id))

        return applied_events + tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CancelFlowCommand):
            return False

        return True
