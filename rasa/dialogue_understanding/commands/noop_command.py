from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from rasa.dialogue_understanding.commands import Command
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class NoopCommand(Command):
    """A command to indicate that nothing needs to be done."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "noop"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NoopCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return NoopCommand()

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

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NoopCommand):
            return False

        return True
