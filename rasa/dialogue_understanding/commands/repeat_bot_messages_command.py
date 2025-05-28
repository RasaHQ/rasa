from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.patterns.repeat import (
    RepeatBotMessagesPatternFlowStackFrame,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class RepeatBotMessagesCommand(Command):
    """A command to indicate that the bot should repeat its last messages."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "repeat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RepeatBotMessagesCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return RepeatBotMessagesCommand()

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        """Runs the command on the tracker.
        Get all the bot utterances until last user utterance and repeat them.

        Args:
            tracker: The tracker to run the command on.
            all_flows: All flows in the assistant.
            original_tracker: The tracker before any command was executed.

        Returns:
            The events to apply to the tracker.
        """
        stack = tracker.stack
        stack.push(RepeatBotMessagesPatternFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RepeatBotMessagesCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "RepeatLastBotMessages()",
            CommandSyntaxVersion.v2: "repeat message",
            CommandSyntaxVersion.v3: "repeat message",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> RepeatBotMessagesCommand:
        """Converts the DSL string to a command."""
        return RepeatBotMessagesCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"RepeatLastBotMessages\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*repeat message['"`]*$""",
            CommandSyntaxVersion.v3: r"""^[\s\W\d]*repeat message['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
