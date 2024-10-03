from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.code_change import CodeChangeFlowStackFrame
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
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

        stack = tracker.stack
        user_frame = top_user_flow_frame(original_tracker.stack)
        current_flow = user_frame.flow(all_flows) if user_frame else None

        if not current_flow:
            structlogger.debug(
                "handle_code_change_command.skip.no_active_flow", command=self
            )
            return []

        stack.push(CodeChangeFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandleCodeChangeCommand):
            return False

        return True
