from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
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
        dialogue_stack = DialogueStack.from_tracker(tracker)
        structlogger.debug("command_executor.error", command=self)
        dialogue_stack.push(HumanHandoffPatternFlowStackFrame())
        return [dialogue_stack.persist_as_event()]
