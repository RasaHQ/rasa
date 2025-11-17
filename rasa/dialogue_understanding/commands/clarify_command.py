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
from rasa.dialogue_understanding.commands.utils import (
    extract_cleaned_options,
    remove_pattern_completed_frames,
)
from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentInterrupted, Event
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
                "clarify_command.altered_command.dropped_clarification_options",
                command=self,
                original_options=self.options,
                cleaned_options=clean_options,
            )
        if len(clean_options) == 0:
            structlogger.debug("clarify_command.empty_clarification", command=self)

        stack = tracker.stack
        relevant_flows = [all_flows.flow_by_id(opt) for opt in clean_options]

        names = [
            flow.readable_name(language=tracker.current_language)
            for flow in relevant_flows
            if flow is not None
        ]

        clarification_ids = [flow.id for flow in relevant_flows if flow is not None]

        applied_events: List[Event] = []

        # if pattern_completed is active, we need to remove it from the stack
        stack, flow_completed_events = remove_pattern_completed_frames(stack)
        applied_events.extend(flow_completed_events)

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

        stack.push(
            ClarifyPatternFlowStackFrame(
                names=names, clarification_ids=clarification_ids
            )
        )
        return applied_events + tracker.create_stack_updated_events(stack)

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
            CommandSyntaxVersion.v3: f"disambiguate flows {' '.join(self.options)}",
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
            CommandSyntaxVersion.v1: r"Clarify\(([\"\'a-zA-Z0-9_, -]*)\)",
            CommandSyntaxVersion.v2: (
                r"""^[\s\W\d]*disambiguate flows (["'a-zA-Z0-9_, -]*)[\W]*$"""
            ),
            CommandSyntaxVersion.v3: (
                r"""^[\s\W\d]*disambiguate flows (["'a-zA-Z0-9_, -]*)[\W]*$"""
            ),
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
