from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.patterns.restart import RestartPatternFlowStackFrame
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class RestartCommand(Command):
    """A command to restart a session."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "restart"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RestartCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return RestartCommand()

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
        stack.push(RestartPatternFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RestartCommand)
