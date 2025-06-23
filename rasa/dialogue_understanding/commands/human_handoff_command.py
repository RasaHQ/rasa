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
        structlogger.debug("human_handoff_command.pushed_to_stack", command=self)
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HumanHandoffCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "HumanHandoff()",
            CommandSyntaxVersion.v2: "hand over",
            CommandSyntaxVersion.v3: "hand over",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> HumanHandoffCommand:
        """Converts the DSL string to a command."""
        return HumanHandoffCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"HumanHandoff\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*hand over['"`]*$""",
            CommandSyntaxVersion.v3: r"""^[\s\W\d]*hand over['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
