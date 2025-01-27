from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.patterns.user_silence import (
    UserSilencePatternFlowStackFrame,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class UserSilenceCommand(Command):
    """A command to indicate user silence."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "user silence"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserSilenceCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return UserSilenceCommand()

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
        stack.push(UserSilencePatternFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UserSilenceCommand)
