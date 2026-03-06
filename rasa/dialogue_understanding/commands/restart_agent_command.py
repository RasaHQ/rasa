from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rasa.core.policies.flows.flow_executor import get_next_step_id_after_agent
from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.patterns.correction import (
    get_suspended_agent_ids_from_stack,
    reset_stack_on_tracker_to_prior_state,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentInterrupted, AgentStarted, Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class RestartAgentCommand(Command):
    """A command to restart an agentic loop within a flow."""

    agent_id: str

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "restart agent"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RestartAgentCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return RestartAgentCommand(agent_id=data["agent_id"])
        except KeyError as e:
            raise ValueError(
                f"Missing parameter '{e}' while parsing RestartAgentCommand."
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

        # get the agent flow and call step (same agent can be called multiple times)
        agent_flow_id = self._get_agent_flow(original_tracker)
        if agent_flow_id is None:
            return []

        agent_frame = original_tracker.stack.find_agent_stack_frame_by_agent(
            self.agent_id
        )
        call_step_id = agent_frame.step_id if agent_frame else None

        step_after_agent = get_next_step_id_after_agent(
            agent_flow_id, self.agent_id, all_flows, tracker, step_id=call_step_id
        )
        if step_after_agent is not None:
            stack = reset_stack_on_tracker_to_prior_state(
                agent_flow_id,
                step_after_agent,
                tracker,
                exclude_agent_id=self.agent_id,
            )

        # create a new agent stack frame to restart the agent (use call_step_id
        # so we use the same call step when the agent is invoked multiple times)
        restart_agent_frame = self.create_restart_agent_stack_frame(
            all_flows, agent_flow_id, step_id=call_step_id
        )

        # if the stack contains an agent stack frame with status
        # "waiting for input" update the status to "interrupted"
        self.update_agent_stack_frames_on_stack(stack)

        # push the stack frame on the top of the stack
        stack.push(restart_agent_frame)
        events: List[Event] = [
            AgentInterrupted(agent_id=agent_id, flow_id=flow_id)
            for agent_id, flow_id in get_suspended_agent_ids_from_stack(stack)
        ]
        events.extend(tracker.create_stack_updated_events(stack))
        return events

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RestartAgentCommand)

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: f"RestartAgent({self.agent_id})",
            CommandSyntaxVersion.v2: f"restart agent {self.agent_id}",
            CommandSyntaxVersion.v3: f"restart agent {self.agent_id}",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> RestartAgentCommand:
        """Converts a DSL string to a command."""
        return RestartAgentCommand(agent_id=str(match.group(1).strip()))

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: r"RestartAgent\(['\"]?([a-zA-Z0-9_-]+)['\"]?\)",
            CommandSyntaxVersion.v2: (
                r"""^[\s\W\d]*restart agent ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
            ),
            CommandSyntaxVersion.v3: (
                r"""^[\s\W\d]*restart agent ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
            ),
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    def create_restart_agent_stack_frame(
        self,
        all_flows: FlowsList,
        agent_flow_id: str,
        step_id: Optional[str] = None,
    ) -> AgentStackFrame:
        """Create a restart agent stack frame for the given flow and call step.

        When step_id is set, use that call step (same agent can be called
        multiple times in one flow). Otherwise use the first matching call step.
        """
        agent_flow = all_flows.flow_by_id(agent_flow_id)
        if not agent_flow:
            raise ValueError(f"Agent flow {agent_flow_id} not found")

        for step in agent_flow.steps:
            if not (isinstance(step, CallFlowStep) and step.call == self.agent_id):
                continue
            if step_id is not None and step.id != step_id:
                continue
            return AgentStackFrame(
                frame_id=f"restart_agent_{self.agent_id}",
                flow_id=agent_flow_id,
                step_id=step.id,
                agent_id=self.agent_id,
                state=AgentState.WAITING_FOR_INPUT,
                is_restart=True,
            )

        raise ValueError(
            f"Call step in agent flow {agent_flow_id} not found"
            + (f" for step_id={step_id!r}" if step_id else "")
        )

    def update_agent_stack_frames_on_stack(self, stack: DialogueStack) -> None:
        for frame in stack.frames:
            if (
                isinstance(frame, AgentStackFrame)
                and frame.state == AgentState.WAITING_FOR_INPUT
            ):
                frame.state = AgentState.INTERRUPTED

    def _get_agent_flow(self, tracker: DialogueStateTracker) -> Optional[str]:
        # find events associated with the agent
        agent_started_events = [
            event
            for event in tracker.events
            if isinstance(event, AgentStarted) and event.agent_id == self.agent_id
        ]
        # take the last one if the agent was started multiple times
        if agent_started_events:
            return agent_started_events[-1].flow_id
        return None
