from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command import Command
from rasa.shared.core.events import Event, SessionEnded
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class SessionEndCommand(Command):
    """A command to indicate the end of a session."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "session end"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionEndCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return SessionEndCommand()

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
        metadata = {"_reason": "user disconnected"}

        # Add metadata sent by the channel connector, if available
        if tracker.latest_message:
            user_metadata = tracker.latest_message.metadata or {}
            metadata.update(user_metadata)

        return [SessionEnded(metadata=metadata)]

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SessionEndCommand)
