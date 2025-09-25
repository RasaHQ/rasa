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
from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentInterrupted, Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class KnowledgeAnswerCommand(FreeFormAnswerCommand):
    """A command to indicate a knowledge-based free-form answer by the bot."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "knowledge"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeAnswerCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return KnowledgeAnswerCommand()

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

        applied_events: List[Event] = []

        # if the top stack frame is an agent stack frame, we need to
        # update the state to INTERRUPTED and add an AgentInterrupted event
        if top_stack_frame := stack.top():
            if isinstance(top_stack_frame, AgentStackFrame):
                applied_events.append(
                    AgentInterrupted(
                        top_stack_frame.agent_id,
                        top_stack_frame.flow_id,
                    )
                )
                top_stack_frame.state = AgentState.INTERRUPTED

        stack.push(SearchPatternFlowStackFrame())
        return applied_events + tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, KnowledgeAnswerCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: "SearchAndReply()",
            CommandSyntaxVersion.v2: "provide info",
            CommandSyntaxVersion.v3: "search and reply",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> KnowledgeAnswerCommand:
        """Converts the DSL string to a command."""
        return KnowledgeAnswerCommand()

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"SearchAndReply\(\)",
            CommandSyntaxVersion.v2: r"""^[\s\W\d]*provide info['"`]*$""",
            CommandSyntaxVersion.v3: r"""^[\s\W\d]*search and reply['"`]*$""",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
