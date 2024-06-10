from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from rasa.dialogue_understanding.commands import Command
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class ChangeFlowCommand(Command):
    """A command to indicate a change of flows was requested by the command
    generator."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "change_flow"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeFlowCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return ChangeFlowCommand()

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        # the change flow command is not actually pushing anything to the tracker,
        # but it is predicted by the MultiStepLLMCommandGenerator and used internally
        return []
