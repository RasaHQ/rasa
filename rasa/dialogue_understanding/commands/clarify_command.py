from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.utils import extract_cleaned_options
from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class ClarifyCommand(Command):
    """A command to indicate that the bot should ask for clarification."""

    options: List[str]

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "clarify"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ClarifyCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return ClarifyCommand(options=data["options"])
        except KeyError as e:
            raise ValueError(
                f"Missing parameter '{e}' while parsing ClarifyCommand."
            ) from e

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
        flows = [all_flows.flow_by_id(opt) for opt in self.options]
        clean_options = [flow.id for flow in flows if flow is not None]
        if len(clean_options) != len(self.options):
            structlogger.debug(
                "command_executor.altered_command.dropped_clarification_options",
                command=self,
                original_options=self.options,
                cleaned_options=clean_options,
            )
        if len(clean_options) == 0:
            structlogger.debug(
                "command_executor.skip_command.empty_clarification", command=self
            )
            return []

        stack = tracker.stack
        relevant_flows = [all_flows.flow_by_id(opt) for opt in clean_options]

        names = [
            flow.readable_name(language=tracker.current_language)
            for flow in relevant_flows
            if flow is not None
        ]

        stack.push(ClarifyPatternFlowStackFrame(names=names))
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(tuple(self.options))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClarifyCommand):
            return False

        return other.options == self.options

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: f"Clarify({', '.join(self.options)})",
            CommandSyntaxVersion.v2: f"disambiguate flows {' '.join(self.options)}",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> Optional[ClarifyCommand]:
        """Converts the DSL string to a command."""
        cleaned_options = extract_cleaned_options(match.group(1))
        return ClarifyCommand(cleaned_options)

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"Clarify\(([\"\'a-zA-Z0-9_, ]*)\)",
            CommandSyntaxVersion.v2: (
                r"""^[\s\W\d]*disambiguate flows (["'a-zA-Z0-9_, ]*)['"`]*$"""
            ),
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
