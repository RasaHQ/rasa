from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.shared.constants import RASA_PATTERN_CANNOT_HANDLE_DEFAULT
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class CannotHandleCommand(Command):
    """A command to indicate that the bot can't handle the user's input."""

    reason: Optional[Text] = RASA_PATTERN_CANNOT_HANDLE_DEFAULT
    """Reason for cannot handle used in switch-case of the
    cannot handle pattern flow."""

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
        return CannotHandleCommand(
            data.get("reason", RASA_PATTERN_CANNOT_HANDLE_DEFAULT)
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
        if self.reason is not None:
            stack.push(CannotHandlePatternFlowStackFrame(reason=self.reason))
        else:
            stack.push(CannotHandlePatternFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.reason)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CannotHandleCommand):
            return False

        return other.reason == self.reason
