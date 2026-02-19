from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _pytest.monkeypatch import MonkeyPatch

from rasa.agents.core.types import ProtocolType
from rasa.agents.exceptions import AgentNameFlowConflictException
from rasa.agents.utils import (
    get_active_agent_info,
    get_agent_info,
    get_completed_agents_info,
    is_agent_completed,
    is_agent_valid,
    resolve_agent_config,
)
from rasa.core.available_agents import ProtocolConfig
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.shared.agents.agent_setup import initialize_agents
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    AgentCompleted,
    AgentInterrupted,
    AgentResumed,
    AgentStarted,
    Event,
)
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker


def make_call_step(
    call: str,
    idx: int = 0,
) -> CallFlowStep:
    """Create a CallFlowStep for testing."""
    return CallFlowStep(
        custom_id=f"call_step_{idx}",
        idx=idx,
        description=f"Test call step for {call}",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call=call,
        mcp_server=None,
        mapping=None,
    )


@pytest_asyncio.fixture
def flows() -> MagicMock:
    mcp_step = make_call_step("agent_mcp")
    a2a_step = make_call_step("agent_a2a", idx=1)

    # Patch the methods to return expected values
    mcp_step.is_calling_mcp_tool = MagicMock(return_value=False)
    mcp_step.is_calling_agent = MagicMock(return_value=True)
    a2a_step.is_calling_mcp_tool = MagicMock(return_value=False)
    a2a_step.is_calling_agent = MagicMock(return_value=True)

    flow = MagicMock()
    flow.steps = [mcp_step, a2a_step]
    flows = MagicMock()
    flows.underlying_flows = [flow]
    flows.flow_by_id = MagicMock(return_value=None)
    return flows


@pytest_asyncio.fixture
def sub_agents(monkeypatch: MonkeyPatch) -> MagicMock:
    mcp_cfg = MagicMock()
    mcp_cfg.configuration = {"foo": "bar"}
    a2a_cfg = MagicMock()
    a2a_cfg.configuration = {"baz": "qux"}
    a2a_cfg.agent.protocol = ProtocolConfig.A2A
    mock_agents = {
        "agent_mcp": mcp_cfg,
        "agent_a2a": a2a_cfg,
    }
    mock_instance = MagicMock()
    mock_instance.agents = mock_agents
    mock_instance.get_agent_config = MagicMock(
        side_effect=lambda agent_name: mock_agents.get(agent_name)
    )

    # Patch Configuration singleton to return our mocked available agents
    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_instance
    monkeypatch.setattr(
        "rasa.core.config.configuration.Configuration.get_instance",
        lambda: mock_configuration_instance,
    )
    return mock_instance


@pytest.mark.asyncio
async def test_initialize_agents_multiple_protocols(
    flows: MagicMock, sub_agents: MagicMock
):
    with patch(
        "rasa.agents.agent_manager.AgentManager.connect_agent", new_callable=AsyncMock
    ) as mock_connect:
        await initialize_agents(flows, sub_agents)
        called_args_list = [call.args for call in mock_connect.await_args_list]
        agent_names = [args[0] for args in called_args_list]
        protocols = [args[1] for args in called_args_list]
        configs = [args[2] for args in called_args_list]

        assert "agent_mcp" in agent_names
        assert "agent_a2a" in agent_names

        for config in configs:
            # Now config is an AgentConfig object, not a dict
            assert hasattr(config, "agent")
            assert hasattr(config, "configuration")
            assert hasattr(config, "connections")
            # Optionally check nested attributes
            assert hasattr(config.agent, "protocol")

        assert ProtocolType.MCP_OPEN in protocols
        assert ProtocolType.A2A in protocols
        assert mock_connect.await_count == 2


@pytest.mark.asyncio
async def test_initialize_agents_validates_flow_conflicts(
    flows: MagicMock, sub_agents: MagicMock
) -> None:
    """Test that agent initialization validates flow name conflicts."""
    # Mock flows to have a conflicting name
    mock_flow = MagicMock()
    mock_flow.id = "conflicting_agent"
    flows.underlying_flows = [mock_flow]

    # Mock sub_agents to have an agent with the same name as a flow
    mock_agent_config = MagicMock()
    mock_agent_config.agent.name = "conflicting_agent"
    sub_agents.agents = {"conflicting_agent": mock_agent_config}

    # Mock the validation function to raise an exception
    with patch(
        "rasa.agents.validation.validate_agent_names_not_conflicting_with_flows"
    ) as mock_validate:
        mock_validate.side_effect = AgentNameFlowConflictException(
            ["conflicting_agent"]
        )

        # Should raise AgentNameFlowConflictException
        with pytest.raises(AgentNameFlowConflictException):
            await initialize_agents(flows, sub_agents)


def test_resolve_agent_config():
    from rasa.core.config.available_endpoints import (
        MCPFromSlotsEntry,
        MCPMetaMapConfig,
        MCPServerConfig,
    )

    server = MagicMock()
    server.name = "server1"
    server.url = None
    server.type = None

    connections = MagicMock()
    connections.mcp_servers = [server]

    agent_config = MagicMock()
    agent_config.connections = connections

    meta_map = MCPMetaMapConfig(
        from_slots=[MCPFromSlotsEntry(slot="user_id", param="user_id")],
    )
    mcp_server_obj = MCPServerConfig(
        name="server1",
        url="http://localhost",
        type="http",
        meta_map=meta_map,
    )
    available_endpoints = MagicMock()
    available_endpoints.mcp_servers = [mcp_server_obj]

    result = resolve_agent_config(agent_config, available_endpoints)

    server_configs = result.connections.mcp_servers
    assert server_configs[0].name == "server1"
    assert server_configs[0].url == "http://localhost"
    assert server_configs[0].type == "http"
    assert server_configs[0].meta_map is meta_map


@pytest.fixture
def mock_agent_config(monkeypatch: MonkeyPatch) -> MagicMock:
    """Fixture for mocking agent config via Configuration.available_agents."""
    mock_available_agents = MagicMock()
    mock_configuration_instance = MagicMock()
    mock_configuration_instance.available_agents = mock_available_agents
    monkeypatch.setattr(
        "rasa.core.config.configuration.Configuration.get_instance",
        lambda: mock_configuration_instance,
    )
    return mock_available_agents.get_agent_config


@pytest.mark.parametrize(
    "agent_id,config_return_value,expected_result",
    [
        ("valid_agent", MagicMock(), True),
        ("invalid_agent", None, False),
    ],
)
def test_is_agent_valid(
    agent_id: str,
    config_return_value: Any,
    expected_result: bool,
    mock_agent_config: MagicMock,
) -> None:
    """Test agent validation with mocked AvailableAgents."""
    mock_agent_config.return_value = config_return_value
    assert is_agent_valid(agent_id) is expected_result


@pytest.fixture
def mock_agent_config_with_info() -> MagicMock:
    """Fixture for mocking agent config with name and description."""
    mock_config = MagicMock()
    mock_config.agent.name = "Test Agent"
    mock_config.agent.description = "A test agent"
    return mock_config


@pytest.mark.parametrize(
    "agent_id,config_return_value,expected_result",
    [
        (
            "test_agent",
            "mock_config",
            {"name": "Test Agent", "description": "A test agent"},
        ),
        ("invalid_agent", None, None),
    ],
)
def test_get_agent_info(
    agent_id: str,
    config_return_value: Any,
    expected_result: Optional[Dict[str, str]],
    mock_agent_config: MagicMock,
    mock_agent_config_with_info: MagicMock,
) -> None:
    """Test agent info retrieval."""
    if config_return_value == "mock_config":
        mock_agent_config.return_value = mock_agent_config_with_info
    else:
        mock_agent_config.return_value = config_return_value

    result = get_agent_info(agent_id)
    assert result == expected_result


@pytest.fixture
def empty_tracker() -> DialogueStateTracker:
    """Fixture for creating an empty tracker."""
    domain = Domain.empty()
    return DialogueStateTracker.from_events("test", [], domain.slots)


@pytest.fixture
def tracker_with_completed_agent() -> DialogueStateTracker:
    """Fixture for creating a tracker with a completed agent."""
    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)
    tracker.update(AgentCompleted("agent1", "flow1"))
    return tracker


@pytest.mark.parametrize(
    "tracker_fixture,agent_id,expected_result",
    [
        ("empty_tracker", "agent1", False),
        ("tracker_with_completed_agent", "agent1", True),
        ("tracker_with_completed_agent", "agent2", False),
    ],
)
def test_is_agent_completed(
    tracker_fixture: str, agent_id: str, expected_result: bool, request: Any
) -> None:
    """Test agent completion checking."""
    tracker = request.getfixturevalue(tracker_fixture)
    assert is_agent_completed(tracker, agent_id) is expected_result


@pytest.mark.parametrize(
    "events,expected_result",
    [
        ([AgentStarted("agent1", "flow1")], False),
        (
            [
                AgentStarted("agent1", "flow1"),
                AgentInterrupted("agent1", "flow1"),
                AgentResumed("agent1", "flow1"),
            ],
            False,
        ),
        (
            [
                AgentStarted("agent1", "flow1"),
                AgentInterrupted("agent1", "flow1"),
                AgentResumed("agent1", "flow1"),
                AgentCompleted("agent1", "flow1"),
            ],
            True,
        ),
        (
            [
                AgentStarted("agent1", "flow1"),
                AgentCompleted("agent1", "flow1"),
                AgentStarted("agent1", "flow1"),
            ],
            False,
        ),
        (
            [
                AgentStarted("agent1", "flow1"),
                AgentCompleted("agent1", "flow1"),
            ],
            True,
        ),
    ],
)
def test_is_agent_completed_with_started_and_resumed_events(
    events: List[Event],
    expected_result: bool,
):
    """Test agent completion checking with started and resumed events."""
    tracker = DialogueStateTracker.from_events("test", events)

    assert is_agent_completed(tracker, "agent1") is expected_result


@pytest.fixture
def tracker_with_multiple_completed_agents() -> DialogueStateTracker:
    """Fixture for creating a tracker with multiple completed agents."""
    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)
    tracker.update(AgentCompleted("agent1", "flow1"))
    tracker.update(AgentCompleted("agent2", "flow2"))
    return tracker


@pytest.fixture
def tracker_multiple_flows_only_active_flow_agents() -> DialogueStateTracker:
    """Test that only agents from the active flow are returned, not from other flows."""
    from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
    from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
        FlowStackFrameType,
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)

    # Add completed agents in different flows
    tracker.update(AgentCompleted("agent1", "flow1"))
    tracker.update(AgentCompleted("agent2", "flow2"))
    tracker.update(AgentCompleted("agent3", "active_flow"))

    # Set active_flow as the currently active flow
    active_flow_frame = UserFlowStackFrame(
        frame_id="active_flow_frame",
        flow_id="active_flow",
        step_id="test_step",
        frame_type=FlowStackFrameType.REGULAR,
    )
    tracker.update_stack(DialogueStack(frames=[active_flow_frame]))
    return tracker


@pytest.fixture
def tracker_flow_restart_agent_running() -> DialogueStateTracker:
    """Test edge case: flow restart with an agent currently running,
    that has completed in a previous flow.
    """
    from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
    from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
        FlowStackFrameType,
    )

    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)

    # First flow execution - agent completed
    tracker.update(AgentCompleted("agent1", "test_flow"))

    # Flow restarted - agent is running again (but not completed yet)
    tracker.update(AgentStarted("agent1", "test_flow"))

    # Set test_flow as the currently active flow
    active_flow_frame = UserFlowStackFrame(
        frame_id="restarted_flow_frame",
        flow_id="test_flow",
        step_id="test_step",
        frame_type=FlowStackFrameType.REGULAR,
    )
    tracker.update_stack(DialogueStack(frames=[active_flow_frame]))
    return tracker


@pytest.mark.parametrize(
    "tracker_fixture,expected_count,expected_agents",
    [
        ("empty_tracker", 0, []),
        (
            "tracker_with_multiple_completed_agents",
            0,  # No active flow, so should return 0 agents
            [],
        ),
        (
            "tracker_multiple_flows_only_active_flow_agents",
            1,  # Should return only 1 agent from active flow, not all 3 from all flows
            [
                {"name": "Agent 3", "description": "Third agent"},
            ],
        ),
        (
            "tracker_flow_restart_agent_running",
            0,  # Should return 0 because agent is currently running / not completed yet
            [],
        ),
    ],
)
def test_get_completed_agents_info(
    tracker_fixture: str,
    expected_count: int,
    expected_agents: List[Dict[str, str]],
    request: Any,
) -> None:
    """Test completed agents info retrieval."""
    tracker = request.getfixturevalue(tracker_fixture)

    if expected_count > 0:
        with patch("rasa.agents.utils.get_agent_info") as mock_get_info:
            mock_get_info.side_effect = expected_agents
            result = get_completed_agents_info(tracker)
    else:
        result = get_completed_agents_info(tracker)

    assert len(result) == expected_count
    if expected_agents:
        assert result == expected_agents


@pytest.fixture
def tracker_with_active_agent() -> DialogueStateTracker:
    """Create a tracker with an active agent in the stack."""
    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)

    # Create a real AgentStackFrame
    active_agent_frame = AgentStackFrame(
        frame_id="active_agent_frame",
        flow_id="test_flow",
        step_id="test_step",
        agent_id="test_agent",
        state=AgentState.WAITING_FOR_INPUT,
    )

    # Update the tracker's stack with the agent frame
    tracker.update_stack(DialogueStack(frames=[active_agent_frame]))
    return tracker


@pytest.fixture
def tracker_with_inactive_agent() -> DialogueStateTracker:
    """Create a tracker with an inactive agent in the stack."""
    domain = Domain.empty()
    tracker = DialogueStateTracker.from_events("test", [], domain.slots)

    # Create an inactive AgentStackFrame
    inactive_agent_frame = AgentStackFrame(
        frame_id="inactive_agent_frame",
        flow_id="test_flow",
        step_id="test_step",
        agent_id="test_agent",
        state=AgentState.INTERRUPTED,  # Not active
    )

    # Update the tracker's stack with the inactive agent frame
    tracker.update_stack(DialogueStack(frames=[inactive_agent_frame]))
    return tracker


@pytest.mark.parametrize(
    "tracker_fixture,flow_id,expected_result",
    [
        ("empty_tracker", "flow1", None),
        ("tracker_with_active_agent", "test_flow", "mock_agent_info"),
        ("tracker_with_active_agent", "wrong_flow", None),
        ("tracker_with_inactive_agent", "test_flow", None),
    ],
)
def test_get_active_agent_info(
    tracker_fixture: str, flow_id: str, expected_result: Optional[str], request: Any
) -> None:
    """Test get_active_agent_info with real tracker and stack objects."""
    tracker = request.getfixturevalue(tracker_fixture)

    if expected_result == "mock_agent_info":
        # Mock get_agent_info to return expected result
        expected_agent_info = {"name": "Test Agent", "description": "A test agent"}
        with patch("rasa.agents.utils.get_agent_info") as mock_get_info:
            mock_get_info.return_value = expected_agent_info
            result = get_active_agent_info(tracker, flow_id)
            assert result == expected_agent_info
            # Verify get_agent_info was called with correct agent_id
            mock_get_info.assert_called_once_with("test_agent")
    else:
        # No agent should be found
        result = get_active_agent_info(tracker, flow_id)
        assert result is None
