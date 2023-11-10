from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class ErrorCommand(Command):
    """A command to indicate that the bot failed to handle the dialogue."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "error"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return ErrorCommand()

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
        structlogger.debug("command_executor.error", command=self)
        stack.push(InternalErrorPatternFlowStackFrame())
        return tracker.create_stack_update_events(stack)
