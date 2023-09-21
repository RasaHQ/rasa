from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from rasa.dialogue_understanding.commands import Command
from rasa.shared.core.events import Event
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class CannotHandleCommand(Command):
    """A command to indicate that the bot can't handle the user's input."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "cannot handle"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CannotHandleCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return CannotHandleCommand()

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
        return []
