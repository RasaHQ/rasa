from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.code_change import CodeChangeFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.dialogue_understanding.stack.utils import top_user_flow_frame

structlogger = structlog.get_logger()


@dataclass
class HandleCodeChangeCommand(Command):
    """A that is executed when the flows have changed."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "handle code change"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandleCodeChangeCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return HandleCodeChangeCommand()

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

        stack = DialogueStack.from_tracker(tracker)
        original_stack = DialogueStack.from_tracker(original_tracker)
        user_frame = top_user_flow_frame(original_stack)
        current_flow = user_frame.flow(all_flows) if user_frame else None

        if not current_flow:
            structlogger.debug(
                "handle_code_change_command.skip.no_active_flow", command=self
            )
            return []

        stack.push(CodeChangeFlowStackFrame())
        return [SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())]
