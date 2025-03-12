from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class ChangeFlowCommand(Command):
    """A command to indicate a change of flows was requested by the command
    generator.
    """

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

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ChangeFlowCommand)

    def __hash__(self) -> int:
        return hash(self.command())

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "ChangeFlow()",
            CommandSyntaxVersion.v2: "change",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @staticmethod
    def from_dsl(match: re.Match, **kwargs: Any) -> ChangeFlowCommand:
        """Converts the DSL string to a command."""
        return ChangeFlowCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"ChangeFlow\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*change['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
