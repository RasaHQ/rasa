import re
from unittest.mock import MagicMock, patch

import pytest

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.restart_agent_command import (
    RestartAgentCommand,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    UserFlowStackFrame,
)
from rasa.shared.core.events import AgentInterrupted, AgentStarted
from rasa.shared.core.flows.flow import Flow
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


@pytest.fixture
def agent_id() -> str:
    return "test_agent"


def test_command_type(agent_id: str):
    assert RestartAgentCommand(agent_id).command() == "restart agent"


def test_from_dict_success(agent_id: str):
    cmd = RestartAgentCommand.from_dict({"agent_id": agent_id})
    assert isinstance(cmd, RestartAgentCommand)
    assert cmd.agent_id == agent_id


def test_from_dict_missing_key():
    with pytest.raises(ValueError):
        RestartAgentCommand.from_dict({})


def test_equality_and_hash(agent_id: str):
    command1 = RestartAgentCommand(agent_id)
    command2 = RestartAgentCommand(agent_id)
    assert command1 == command2
    assert hash(command1) == hash(command2)
    assert command1 == command1
    assert not (command1 != command2)
    assert command1 != object()


@pytest.mark.parametrize(
    "version",
    [
        CommandSyntaxVersion.v1,
        CommandSyntaxVersion.v2,
        CommandSyntaxVersion.v3,
    ],
)
def test_to_dsl_versions(agent_id: str, version: CommandSyntaxVersion):
    cmd = RestartAgentCommand(agent_id)
    with patch.object(CommandSyntaxManager, "get_syntax_version", return_value=version):
        dsl = cmd.to_dsl()
        assert agent_id in dsl


@pytest.mark.parametrize(
    "version, dsl_string",
    [
        (CommandSyntaxVersion.v1, "RestartAgent(test_agent)"),
        (CommandSyntaxVersion.v2, "restart agent test_agent"),
        (CommandSyntaxVersion.v3, "restart agent test_agent"),
    ],
)
def test_from_dsl(agent_id: str, version: CommandSyntaxVersion, dsl_string: str):
    with patch.object(CommandSyntaxManager, "get_syntax_version", return_value=version):
        pattern = RestartAgentCommand.regex_pattern()
        match = re.match(pattern, dsl_string)
        assert match is not None
        cmd = RestartAgentCommand.from_dsl(match)
        assert cmd.agent_id == agent_id


@pytest.mark.parametrize(
    "version, command_string",
    [
        (CommandSyntaxVersion.v1, "RestartAgent(test_agent)"),
        (CommandSyntaxVersion.v2, "restart agent test_agent"),
        (CommandSyntaxVersion.v3, "restart agent test_agent"),
    ],
)
def test_regex_pattern_versions(
    agent_id: str, version: CommandSyntaxVersion, command_string: str
):
    with patch.object(CommandSyntaxManager, "get_syntax_version", return_value=version):
        pattern = RestartAgentCommand.regex_pattern()
        assert isinstance(pattern, str)
        assert re.match(pattern, command_string)


def test_create_restart_agent_stack_frame(agent_id: str):
    # mock all_flows with a flow that has a call step with the agent id
    call_step = MagicMock(spec=CallFlowStep)
    call_step.call = agent_id
    call_step.id = "first_step"
    call_step.next = None

    flow_mock = MagicMock(spec=Flow)
    flow_mock.id = "flow_123"
    flow_mock.steps = [call_step]

    all_flows = MagicMock()
    all_flows.flow_by_id.return_value = flow_mock

    cmd = RestartAgentCommand(agent_id)
    frame = cmd.create_restart_agent_stack_frame(all_flows, "flow_123")
    assert isinstance(frame, AgentStackFrame)
    assert frame.agent_id == agent_id
    assert frame.state == AgentState.WAITING_FOR_INPUT
    assert frame.flow_id == "flow_123"
    assert frame.step_id == "first_step"


def test_create_restart_agent_stack_frame_uses_step_id_when_agent_called_twice():
    """step_id selects the correct call step when the same agent is called twice."""
    first_call = MagicMock(spec=CallFlowStep)
    first_call.call = "my_agent"
    first_call.id = "first_call_step"
    first_call.next = None
    second_call = MagicMock(spec=CallFlowStep)
    second_call.call = "my_agent"
    second_call.id = "second_call_step"
    second_call.next = None

    flow_mock = MagicMock(spec=Flow)
    flow_mock.id = "flow_123"
    flow_mock.steps = [first_call, second_call]

    all_flows = MagicMock()
    all_flows.flow_by_id.return_value = flow_mock

    cmd = RestartAgentCommand("my_agent")
    frame = cmd.create_restart_agent_stack_frame(
        all_flows, "flow_123", step_id="second_call_step"
    )
    assert frame.step_id == "second_call_step"


def test_update_agent_stack_frames_on_stack(agent_id: str):
    cmd = RestartAgentCommand(agent_id)
    frame1 = AgentStackFrame("f1", "flow1", agent_id, AgentState.WAITING_FOR_INPUT)
    frame2 = AgentStackFrame("f2", "flow2", agent_id, AgentState.INTERRUPTED)
    stack = MagicMock()
    stack.frames = [frame1, frame2]
    cmd.update_agent_stack_frames_on_stack(stack)
    assert frame1.state == AgentState.INTERRUPTED
    assert frame2.state == AgentState.INTERRUPTED


def test_get_agent_flow(agent_id: str):
    cmd = RestartAgentCommand(agent_id)
    flow_id = "flow_abc"
    event = AgentStarted(agent_id=agent_id, flow_id=flow_id)
    tracker = MagicMock()
    tracker.events = [event]
    assert cmd._get_agent_flow(tracker) == flow_id

    # Should return None if no matching event
    tracker.events = []
    assert cmd._get_agent_flow(tracker) is None


def test_run_command_on_tracker(agent_id: str):
    # mock all_flows with a flow that has a call step with the agent id
    call_step = MagicMock(spec=CallFlowStep)
    call_step.call = agent_id
    call_step.id = "first_step"
    call_step.next = None

    flow_mock = MagicMock(spec=Flow)
    flow_mock.id = "flow_abc"
    flow_mock.steps = [call_step]

    all_flows = MagicMock()
    all_flows.flow_by_id.return_value = flow_mock

    cmd = RestartAgentCommand(agent_id)
    flow_id = "flow_abc"
    agent_started_event = AgentStarted(agent_id=agent_id, flow_id=flow_id)
    tracker = MagicMock()
    tracker.stack = DialogueStack([])
    tracker.create_stack_updated_events.return_value = ["stack_updated"]
    original_tracker = MagicMock()
    original_tracker.events = [agent_started_event]
    # Empty stack so find_agent_stack_frame_by_agent returns None (call_step_id=None)
    original_tracker.stack = DialogueStack([])

    result = cmd.run_command_on_tracker(tracker, all_flows, original_tracker)
    assert result == ["stack_updated"]

    # Check that the top frame on the stack has the correct flow_id
    top_frame = tracker.stack.frames[-1]
    assert isinstance(top_frame, AgentStackFrame)
    assert top_frame.flow_id == flow_id
    assert top_frame.step_id == call_step.id


def test_restart_agent_command_emits_agent_interrupted_for_suspended_agents():
    """ENG-2669: When restart suspends other agents, AgentInterrupted is emitted."""
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: flow for agent restart test
            steps:
            - id: call_agent_a
              call: agent_a
            - id: step_between
              collect: slot_x
            - id: call_agent_b
              call: agent_b
          agent_a:
            description: agent a
            steps:
            - id: step_a
              action: action_listen
          agent_b:
            description: agent b
            steps:
            - id: step_b
              action: action_listen
        """
    )
    flow_id = "my_flow"
    agent_a = "agent_a"
    agent_b = "agent_b"
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update(AgentStarted(agent_id=agent_a, flow_id=flow_id))
    user_flow = UserFlowStackFrame(
        flow_id=flow_id,
        step_id="step_between",
        frame_id="uf-1",
    )
    tracker.update_stack(DialogueStack(frames=[user_flow]))
    agent_b_frame = AgentStackFrame(
        flow_id=flow_id,
        step_id="call_agent_b",
        agent_id=agent_b,
        state=AgentState.INTERRUPTED,
        frame_id="agent-b-frame",
    )
    tracker.update_stack(DialogueStack(frames=[user_flow, agent_b_frame]))
    original_tracker = DialogueStateTracker.from_events(
        "test", evts=[AgentStarted(agent_id=agent_a, flow_id=flow_id)]
    )
    cmd = RestartAgentCommand(agent_a)
    events = cmd.run_command_on_tracker(tracker, all_flows, original_tracker)
    interrupted = [e for e in events if isinstance(e, AgentInterrupted)]
    assert len(interrupted) == 1
    assert interrupted[0].agent_id == agent_b
    assert interrupted[0].flow_id == flow_id
