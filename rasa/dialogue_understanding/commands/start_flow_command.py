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
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    top_user_flow_frame,
    user_flows_on_the_stack,
)
from rasa.shared.core.events import Event, FlowInterrupted
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class StartFlowCommand(Command):
    """A command to start a flow."""

    flow: str

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "start flow"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StartFlowCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return StartFlowCommand(flow=data["flow"])
        except KeyError as e:
            raise ValueError(
                f"Missing parameter '{e}' while parsing StartFlowCommand."
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
        stack = tracker.stack
        original_stack = original_tracker.stack
        applied_events: List[Event] = []

        if self.flow in user_flows_on_the_stack(stack):
            structlogger.debug(
                "start_flow_command.skip_command.already_started_flow", command=self
            )
            return []
        elif self.flow not in all_flows.flow_ids:
            structlogger.debug(
                "start_flow_command.skip_command.start_invalid_flow_id", command=self
            )
            return []

        original_user_frame = top_user_flow_frame(original_stack)
        original_top_flow = (
            original_user_frame.flow(all_flows) if original_user_frame else None
        )

        frame_type = FlowStackFrameType.REGULAR

        if original_top_flow:
            frame_type = FlowStackFrameType.INTERRUPT

            if original_user_frame is not None:
                applied_events.append(
                    FlowInterrupted(
                        original_user_frame.flow_id, original_user_frame.step_id
                    )
                )

        structlogger.debug("start_flow_command.start_flow", command=self)
        stack.push(UserFlowStackFrame(flow_id=self.flow, frame_type=frame_type))
        return applied_events + tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.flow)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StartFlowCommand):
            return False

        return other.flow == self.flow

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: f"StartFlow({self.flow})",
            CommandSyntaxVersion.v2: f"start flow {self.flow}",
            CommandSyntaxVersion.v3: f"start flow {self.flow}",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> Optional[StartFlowCommand]:
        """Converts the DSL string to a command."""
        return StartFlowCommand(flow=str(match.group(1).strip()))

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"StartFlow\(['\"]?([a-zA-Z0-9_-]+)['\"]?\)",
            CommandSyntaxVersion.v2: (
                r"""^[\s\W\d]*start flow ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
            ),
            CommandSyntaxVersion.v3: (
                r"""^[\s\W\d]*start flow ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
            ),
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
