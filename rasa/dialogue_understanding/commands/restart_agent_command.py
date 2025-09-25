from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.core.events import AgentStarted, Event
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

        # get the agent flow
        agent_flow_id = self._get_agent_flow(original_tracker)

        # create a new agent stack frame to restart the agent
        restart_agent_frame = self.create_restart_agent_stack_frame(
            all_flows, agent_flow_id
        )

        # if the stack contains an agent stack frame with status
        # "waiting for input" update the status to "interrupted"
        self.update_agent_stack_frames_on_stack(stack)

        # push the stack frame on the top of the stack
        stack.push(restart_agent_frame)
        return tracker.create_stack_updated_events(stack)

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
        self, all_flows: FlowsList, agent_flow_id: str
    ) -> AgentStackFrame:
        # get the agent flow
        agent_flow = all_flows.flow_by_id(agent_flow_id)
        if not agent_flow:
            raise ValueError(f"Agent flow {agent_flow_id} not found")

        # find the call flow step for the agent
        # set the state to "waiting for input" so that the agent can process
        # the latest user message
        for step in agent_flow.steps:
            if isinstance(step, CallFlowStep) and step.call == self.agent_id:
                return AgentStackFrame(
                    frame_id=f"restart_agent_{self.agent_id}",
                    flow_id=agent_flow_id,
                    step_id=step.id,
                    agent_id=self.agent_id,
                    state=AgentState.WAITING_FOR_INPUT,
                )

        raise ValueError(f"Call step in agent flow {agent_flow_id} not found")

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
            if type(event) == AgentStarted and event.agent_id == self.agent_id
        ]
        # take the last one if the agent was started multiple times
        if agent_started_events:
            return agent_started_events[-1].flow_id
        return None
