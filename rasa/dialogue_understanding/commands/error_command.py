from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Text

import structlog

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.shared.constants import RASA_PATTERN_INTERNAL_ERROR_DEFAULT
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class ErrorCommand(Command):
    """A command to indicate that the bot failed to handle the dialogue."""

    error_type: Text = RASA_PATTERN_INTERNAL_ERROR_DEFAULT
    info: Dict[Text, Any] = field(default_factory=dict)

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
        return ErrorCommand(
            error_type=data.get("error_type", RASA_PATTERN_INTERNAL_ERROR_DEFAULT),
            info=data.get("info", {}),
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
        structlogger.debug("error_command.error", command=self)
        stack.push(
            InternalErrorPatternFlowStackFrame(
                error_type=self.error_type, info=self.info
            )
        )
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        hashed = hash(self.error_type)
        if self.info:
            hashed += hash(str(self.info))
        return hashed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ErrorCommand):
            return False

        return other.error_type == self.error_type and other.info == self.info
