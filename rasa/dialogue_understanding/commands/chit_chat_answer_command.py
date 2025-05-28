from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.free_form_answer_command import (
    FreeFormAnswerCommand,
)
from rasa.dialogue_understanding.patterns.chitchat import ChitchatPatternFlowStackFrame
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class ChitChatAnswerCommand(FreeFormAnswerCommand):
    """A command to indicate a chitchat style free-form answer by the bot."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "chitchat"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChitChatAnswerCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return ChitChatAnswerCommand()

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
        stack.push(ChitchatPatternFlowStackFrame())
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChitChatAnswerCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "ChitChat()",
            CommandSyntaxVersion.v2: "offtopic reply",
            CommandSyntaxVersion.v3: "offtopic reply",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> ChitChatAnswerCommand:
        """Converts the DSL string to a command."""
        return ChitChatAnswerCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"ChitChat\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*offtopic reply['"`]*$""",
            CommandSyntaxVersion.v3: r"""^[\s\W\d]*offtopic reply['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
