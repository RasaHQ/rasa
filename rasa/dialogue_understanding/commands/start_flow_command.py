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
    remove_pattern_completed_frames,
    remove_pattern_continue_interrupted_frames,
    resume_flow,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentState,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    is_pattern_active,
    top_user_flow_frame,
    user_flows_on_the_stack,
)
from rasa.shared.core.events import (
    AgentInterrupted,
    Event,
    FlowInterrupted,
)
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

        if self.flow not in all_flows.flow_ids:
            structlogger.debug(
                "start_flow_command.skip_command.start_invalid_flow_id", command=self
            )
            return []

        original_user_frame = top_user_flow_frame(original_stack)
        original_top_flow = (
            original_user_frame.flow(all_flows) if original_user_frame else None
        )

        # if pattern_completed is active, we need to remove it from the stack
        stack, flow_completed_events = remove_pattern_completed_frames(stack)
        applied_events.extend(flow_completed_events)

        # if the original top flow is the same as the flow to start, the flow is
        # already active, do nothing
        if original_top_flow is not None and original_top_flow.id == self.flow:
            # in case continue_interrupted is not active, skip the already active start
            # flow command
            if not is_pattern_active(stack, ContinueInterruptedPatternFlowStackFrame):
                return applied_events + tracker.create_stack_updated_events(stack)

            # if the continue interrupted flow is active, and the command generator
            # predicted a start flow command for the flow which is on top of the stack,
            # we just need to remove the pattern_continue_interrupted frame(s) from the
            # stack
            stack, flow_completed_events = remove_pattern_continue_interrupted_frames(
                stack
            )
            applied_events.extend(flow_completed_events)
            return applied_events + tracker.create_stack_updated_events(stack)

        # if the flow is already on the stack, resume it
        if (
            self.flow in user_flows_on_the_stack(stack)
            and original_user_frame is not None
        ):
            # if pattern_continue_interrupted is active, we need to remove it
            # from the stack before resuming the flow
            stack, flow_completed_events = remove_pattern_continue_interrupted_frames(
                stack
            )
            applied_events.extend(flow_completed_events)
            applied_events.extend(resume_flow(self.flow, tracker, stack))
            # the current active flow is interrupted
            applied_events.append(
                FlowInterrupted(
                    original_user_frame.flow_id, original_user_frame.step_id
                )
            )
            return applied_events

        frame_type = FlowStackFrameType.REGULAR

        # remove the pattern_continue_interrupted frames from the stack
        # if it is currently active but the user digressed from the pattern
        stack, flow_completed_events = remove_pattern_continue_interrupted_frames(stack)
        applied_events.extend(flow_completed_events)

        if original_top_flow:
            # if the original top flow is not the same as the flow to start,
            # interrupt the current active flow
            frame_type = FlowStackFrameType.INTERRUPT

            if original_user_frame is not None:
                applied_events.append(
                    FlowInterrupted(
                        original_user_frame.flow_id, original_user_frame.step_id
                    )
                )

            # If there is an active agent frame, interrupt it
            active_agent_stack_frame = stack.find_active_agent_frame()
            if active_agent_stack_frame:
                structlogger.debug(
                    "start_flow_command.interrupt_agent",
                    command=self,
                    agent_id=active_agent_stack_frame.agent_id,
                    frame_id=active_agent_stack_frame.frame_id,
                    flow_id=active_agent_stack_frame.flow_id,
                )
                active_agent_stack_frame.state = AgentState.INTERRUPTED
                applied_events.append(
                    AgentInterrupted(
                        active_agent_stack_frame.agent_id,
                        active_agent_stack_frame.flow_id,
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
