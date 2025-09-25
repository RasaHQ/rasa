from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class ContinueAgentCommand(Command):
    """A command to continue the currently active agent's execution."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "continue agent"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContinueAgentCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return ContinueAgentCommand()

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
        # do nothing
        return []

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContinueAgentCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "ContinueAgent()",
            CommandSyntaxVersion.v2: "continue agent",
            CommandSyntaxVersion.v3: "continue agent",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> ContinueAgentCommand:
        """Converts a DSL string to a command."""
        return ContinueAgentCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"ContinueAgent\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*continue agent['"`]*$""",
            CommandSyntaxVersion.v3: r"""^[\s\W\d]*continue agent['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
