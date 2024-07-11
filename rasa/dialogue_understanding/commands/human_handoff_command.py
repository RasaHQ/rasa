from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class HumanHandoffCommand(Command):
    """A command to indicate that the bot should handoff to a human."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "human handoff"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HumanHandoffCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return HumanHandoffCommand()

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
        stack.push(HumanHandoffPatternFlowStackFrame())
        structlogger.debug(
            "command_executor.human_handoff.pushed_to_stack", command=self
        )
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HumanHandoffCommand):
            return False

        return True
