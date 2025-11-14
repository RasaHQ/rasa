"""Unit tests for AgentManager."""

from unittest.mock import AsyncMock, patch

import pytest

from rasa.agents.agent_manager import AgentManager
from rasa.agents.core.types import AgentIdentifier, AgentStatus, ProtocolType
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentOutput
from rasa.core.available_agents import AgentConfig, AgentInfo, ProtocolConfig
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.core.events import SlotSet
from rasa.shared.exceptions import AgentInitializationException


@pytest.fixture
def agent_manager() -> AgentManager:
    """Fixture for creating a fresh AgentManager instance."""
    manager = AgentManager()
    # Clear the agents dictionary to ensure clean state
    manager.agents.clear()
    return manager


@pytest.fixture
def mock_agent_protocol() -> AsyncMock:
    """Fixture for creating a mock AgentProtocol."""
    mock_agent = AsyncMock()
    mock_agent.protocol_type = ProtocolType.MCP_TASK
    mock_agent.connect = AsyncMock()
    mock_agent.disconnect = AsyncMock()
    mock_agent.process_input = AsyncMock()
    mock_agent.run = AsyncMock()
    mock_agent.process_output = AsyncMock()
    return mock_agent


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Fixture for creating a mock AgentConfig."""
    return AgentConfig(
        agent=AgentInfo(
            name="test_agent",
            protocol=ProtocolConfig.RASA,
            description="A test agent for unit testing",
        )
    )


@pytest.fixture
def mock_agent_input() -> AgentInput:
    """Fixture for creating a mock AgentInput."""
    return AgentInput(
        id="test_id",
        user_message="Hello, how can you help me?",
        slots=[
            AgentInputSlot(
                name="user_name", value="John", type="text", allowed_values=None
            )
        ],
        conversation_history="Previous conversation...",
        events=[],
        metadata={"key": "value"},
        timestamp="2024-01-15T10:30:00Z",
    )


@pytest.fixture
def mock_agent_output() -> AgentOutput:
    """Fixture for creating a mock AgentOutput."""
    return AgentOutput(
        id="test_id",
        status=AgentStatus.COMPLETED,
        response_message="Hello! I can help you with various tasks.",
        events=[SlotSet("user_name", "John")],
        structured_results=None,
        metadata={"processed": True},
        timestamp="2024-01-15T10:30:01Z",
        error_message=None,
    )


def test_singleton_behavior() -> None:
    """Test that AgentManager follows singleton pattern."""
    # Clear any existing instance
    AgentManager._instance = None

    # Create two instances
    manager1 = AgentManager()
    manager2 = AgentManager()

    # They should be the same instance
    assert manager1 is manager2
    assert id(manager1) == id(manager2)

    # Clean up
    manager1.agents.clear()


def test_initial_state(agent_manager: AgentManager) -> None:
    """Test that AgentManager starts with empty agents dictionary."""
    assert agent_manager.agents == {}


def test_add_agent_success(
    agent_manager: AgentManager, mock_agent_protocol: AsyncMock
) -> None:
    """Test successful agent addition."""
    agent_identifier = make_agent_identifier("add_test_agent", ProtocolType.MCP_TASK)

    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    assert agent_identifier in agent_manager.agents
    assert agent_manager.agents[agent_identifier] == mock_agent_protocol


def test_add_agent_duplicate_raises_error(
    agent_manager: AgentManager, mock_agent_protocol: AsyncMock
) -> None:
    """Test that adding duplicate agent raises ValueError."""
    agent_identifier = make_agent_identifier(
        "duplicate_test_agent", ProtocolType.MCP_TASK
    )

    # Add agent first time
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Try to add same agent again
    with pytest.raises(
        ValueError, match="Agent duplicate_test_agent::mcp_task already exists"
    ):
        agent_manager._add_agent(agent_identifier, mock_agent_protocol)


def test_get_agent_success(
    agent_manager: AgentManager, mock_agent_protocol: AsyncMock
) -> None:
    """Test successful agent retrieval."""
    agent_identifier = make_agent_identifier("get_test_agent", ProtocolType.MCP_TASK)

    # Add agent first
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Get agent
    retrieved_agent = agent_manager.get_agent("get_test_agent", ProtocolType.MCP_TASK)
    assert retrieved_agent == mock_agent_protocol


def test_get_agent_not_found_raises_error(agent_manager: AgentManager) -> None:
    """Test that getting non-existent agent raises ValueError."""
    with pytest.raises(
        ValueError, match="Agent non_existent_get::mcp_task is not available"
    ):
        agent_manager.get_agent("non_existent_get", ProtocolType.MCP_TASK)


@pytest.mark.asyncio
async def test_connect_agent_success(
    agent_manager: AgentManager, mock_agent_config: AgentConfig
) -> None:
    """Test successful agent connection."""
    with patch("rasa.agents.agent_manager.AgentFactory") as mock_factory:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_factory.create_client.return_value = mock_client

        await agent_manager.connect_agent(
            "connect_success_agent", ProtocolType.MCP_TASK, mock_agent_config
        )

        # Verify factory was called correctly
        mock_factory.create_client.assert_called_once_with(
            ProtocolType.MCP_TASK, mock_agent_config
        )

        # Verify client connect was called
        mock_client.connect.assert_called_once()

        # Verify agent was added to manager
        agent_identifier = make_agent_identifier(
            "connect_success_agent", ProtocolType.MCP_TASK
        )
        assert agent_identifier in agent_manager.agents
        assert agent_manager.agents[agent_identifier] == mock_client


@pytest.mark.asyncio
async def test_connect_agent_factory_failure(
    agent_manager: AgentManager, mock_agent_config: AgentConfig
) -> None:
    """Test agent connection failure during factory creation."""
    # Ensure the agent is not already in the manager
    agent_identifier = make_agent_identifier(
        "connect_factory_fail_agent", ProtocolType.MCP_TASK
    )
    assert agent_identifier not in agent_manager.agents

    with patch("rasa.agents.agent_manager.AgentFactory") as mock_factory:
        mock_factory.create_client.side_effect = Exception("Factory error")

        with pytest.raises(AgentInitializationException) as exc_info:
            await agent_manager.connect_agent(
                "connect_factory_fail_agent",
                ProtocolType.MCP_TASK,
                mock_agent_config,
            )

        assert "Factory error" in str(exc_info.value)

        # Verify agent was not added to manager
        assert agent_identifier not in agent_manager.agents


@pytest.mark.asyncio
async def test_connect_agent_connect_failure(
    agent_manager: AgentManager, mock_agent_config: AgentConfig
) -> None:
    """Test agent connection failure during client connect."""
    # Ensure the agent is not already in the manager
    agent_identifier = make_agent_identifier(
        "connect_connect_fail_agent", ProtocolType.MCP_TASK
    )
    assert agent_identifier not in agent_manager.agents

    with patch("rasa.agents.agent_manager.AgentFactory") as mock_factory:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock(side_effect=Exception("Connection error"))
        mock_factory.create_client.return_value = mock_client

        with pytest.raises(AgentInitializationException) as exc_info:
            await agent_manager.connect_agent(
                "connect_connect_fail_agent",
                ProtocolType.MCP_TASK,
                mock_agent_config,
            )

        assert "Connection error" in str(exc_info.value)

        # Verify agent was not added to manager
        assert agent_identifier not in agent_manager.agents


@pytest.mark.asyncio
async def test_connect_agent_already_connected(
    agent_manager: AgentManager, mock_agent_config: AgentConfig
) -> None:
    """Test connecting agent that is already connected."""
    agent_identifier = make_agent_identifier(
        "already_connected_agent", ProtocolType.MCP_TASK
    )
    mock_existing_agent = AsyncMock()
    agent_manager._add_agent(agent_identifier, mock_existing_agent)

    with (
        patch("rasa.agents.agent_manager.AgentFactory") as mock_factory,
        patch("rasa.agents.agent_manager.structlogger") as mock_logger,
    ):
        # The method should return early without calling factory or connecting
        await agent_manager.connect_agent(
            "already_connected_agent", ProtocolType.MCP_TASK, mock_agent_config
        )

        # Verify factory was not called
        mock_factory.create_client.assert_not_called()

        # Verify info logging about already connected
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args
        assert "agent_manager.connect_agent.already_connected" in call_args[0][0]
        assert "agent_id" in call_args[1]
        assert "agent_name" in call_args[1]
        assert "already connected" in call_args[1]["event_info"]

        # Verify the existing agent is still in the manager
        assert agent_identifier in agent_manager.agents
        assert agent_manager.agents[agent_identifier] == mock_existing_agent


@pytest.mark.asyncio
async def test_run_agent_success(
    agent_manager: AgentManager,
    mock_agent_protocol: AsyncMock,
    mock_agent_input: AgentInput,
    mock_agent_output: AgentOutput,
) -> None:
    """Test successful agent execution."""
    agent_identifier = make_agent_identifier("run_success_agent", ProtocolType.MCP_TASK)
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Mock the agent methods
    processed_input = AgentInput(**mock_agent_input.dict())
    mock_agent_protocol.process_input.return_value = processed_input
    mock_agent_protocol.run.return_value = mock_agent_output
    mock_agent_protocol.process_output.return_value = mock_agent_output

    result = await agent_manager.run_agent(
        "run_success_agent", ProtocolType.MCP_TASK, mock_agent_input
    )

    # Verify all agent methods were called
    mock_agent_protocol.process_input.assert_called_once_with(mock_agent_input)
    mock_agent_protocol.run.assert_called_once_with(
        processed_input, output_channel=None
    )
    mock_agent_protocol.process_output.assert_called_once_with(mock_agent_output)

    # Verify result
    assert result == mock_agent_output


@pytest.mark.asyncio
async def test_run_agent_not_found(
    agent_manager: AgentManager, mock_agent_input: AgentInput
) -> None:
    """Test running agent that doesn't exist."""
    with pytest.raises(
        ValueError, match="Agent non_existent_run::mcp_task is not available"
    ):
        await agent_manager.run_agent(
            "non_existent_run", ProtocolType.MCP_TASK, mock_agent_input
        )


@pytest.mark.asyncio
async def test_run_agent_run_failure(
    agent_manager: AgentManager,
    mock_agent_protocol: AsyncMock,
    mock_agent_input: AgentInput,
) -> None:
    """Test agent execution failure during run."""
    agent_identifier = make_agent_identifier(
        "run_run_fail_agent", ProtocolType.MCP_TASK
    )
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Mock process_input to succeed but run to fail
    processed_input = AgentInput(**mock_agent_input.dict())
    mock_agent_protocol.process_input.return_value = processed_input
    mock_agent_protocol.run.side_effect = Exception("Run error")

    with pytest.raises(Exception, match="Run error"):
        await agent_manager.run_agent(
            "run_run_fail_agent", ProtocolType.MCP_TASK, mock_agent_input
        )


async def test_run_agent_process_input_failure(
    agent_manager: AgentManager,
    mock_agent_protocol: AsyncMock,
    mock_agent_input: AgentInput,
) -> None:
    """Test agent execution failure during process_input."""
    agent_identifier = make_agent_identifier(
        "run_process_input_fail_agent", ProtocolType.MCP_TASK
    )
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Mock process_input to fail
    mock_agent_protocol.process_input.side_effect = Exception("Process input error")

    with pytest.raises(Exception, match="Process input error"):
        await agent_manager.run_agent(
            "run_process_input_fail_agent", ProtocolType.MCP_TASK, mock_agent_input
        )


async def test_run_agent_process_output_failure(
    agent_manager: AgentManager,
    mock_agent_protocol: AsyncMock,
    mock_agent_output: AgentOutput,
) -> None:
    """Test agent execution failure during process_output."""
    agent_identifier = make_agent_identifier(
        "run_process_output_fail_agent", ProtocolType.MCP_TASK
    )
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Mock process_output to fail
    mock_agent_protocol.process_output.side_effect = Exception("Process output error")

    with pytest.raises(Exception, match="Process output error"):
        await agent_manager.run_agent(
            "run_process_output_fail_agent", ProtocolType.MCP_TASK, mock_agent_output
        )


@pytest.mark.asyncio
async def test_disconnect_agent_success(
    agent_manager: AgentManager, mock_agent_protocol: AsyncMock
) -> None:
    """Test successful agent disconnection."""
    agent_identifier = make_agent_identifier(
        "disconnect_success_agent", ProtocolType.MCP_TASK
    )
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    await agent_manager.disconnect_agent(
        "disconnect_success_agent", ProtocolType.MCP_TASK
    )

    # Verify disconnect was called
    mock_agent_protocol.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_agent_not_found(agent_manager: AgentManager) -> None:
    """Test disconnecting agent that doesn't exist."""
    with pytest.raises(
        ValueError, match="Agent `non_existent_disconnect::mcp_task` is not available"
    ):
        await agent_manager.disconnect_agent(
            "non_existent_disconnect", ProtocolType.MCP_TASK
        )


@pytest.mark.asyncio
async def test_disconnect_agent_disconnect_failure(
    agent_manager: AgentManager, mock_agent_protocol: AsyncMock
) -> None:
    """Test agent disconnection failure during disconnect."""
    agent_identifier = make_agent_identifier(
        "disconnect_fail_agent", ProtocolType.MCP_TASK
    )
    agent_manager._add_agent(agent_identifier, mock_agent_protocol)

    # Mock disconnect to raise exception
    mock_agent_protocol.disconnect.side_effect = Exception("Disconnect error")

    with pytest.raises(ConnectionError) as exc_info:
        await agent_manager.disconnect_agent(
            "disconnect_fail_agent", ProtocolType.MCP_TASK
        )

    assert "Disconnect error" in str(exc_info.value)

    # Verify agent was not removed from manager due to failure
    assert agent_identifier in agent_manager.agents


def test_agent_identifier_creation() -> None:
    """Test that agent identifiers are created correctly."""
    identifier = make_agent_identifier("test_agent", ProtocolType.MCP_TASK)
    expected = AgentIdentifier("test_agent", ProtocolType.MCP_TASK)
    assert identifier == expected
    assert str(identifier) == "test_agent::mcp_task"


def test_multiple_agents_management(agent_manager: AgentManager) -> None:
    """Test managing multiple agents with different protocols."""
    # Add multiple agents
    agent1_id = make_agent_identifier("agent1", ProtocolType.MCP_TASK)
    agent2_id = make_agent_identifier("agent2", ProtocolType.MCP_OPEN)
    agent3_id = make_agent_identifier("agent3", ProtocolType.A2A)

    mock_agent1 = AsyncMock()
    mock_agent2 = AsyncMock()
    mock_agent3 = AsyncMock()

    agent_manager._add_agent(agent1_id, mock_agent1)
    agent_manager._add_agent(agent2_id, mock_agent2)
    agent_manager._add_agent(agent3_id, mock_agent3)

    # Verify all agents are stored
    assert len(agent_manager.agents) == 3
    assert agent1_id in agent_manager.agents
    assert agent2_id in agent_manager.agents
    assert agent3_id in agent_manager.agents

    # Verify correct agents are retrieved
    assert agent_manager.get_agent("agent1", ProtocolType.MCP_TASK) == mock_agent1
    assert agent_manager.get_agent("agent2", ProtocolType.MCP_OPEN) == mock_agent2
    assert agent_manager.get_agent("agent3", ProtocolType.A2A) == mock_agent3


def test_agent_with_same_name_different_protocols(agent_manager: AgentManager) -> None:
    """Test that agents with same name but different protocols
    are separately managed.
    """
    agent1_id = make_agent_identifier("same_name", ProtocolType.MCP_TASK)
    agent2_id = make_agent_identifier("same_name", ProtocolType.MCP_OPEN)

    mock_agent1 = AsyncMock()
    mock_agent2 = AsyncMock()

    # Both should be addable
    agent_manager._add_agent(agent1_id, mock_agent1)
    agent_manager._add_agent(agent2_id, mock_agent2)

    assert len(agent_manager.agents) == 2
    assert agent_manager.get_agent("same_name", ProtocolType.MCP_TASK) == mock_agent1
    assert agent_manager.get_agent("same_name", ProtocolType.MCP_OPEN) == mock_agent2


@pytest.mark.asyncio
async def test_agent_manager_with_identical_agent_names(
    mock_agent_config: AgentConfig, agent_manager: AgentManager
) -> None:
    """Test that AgentManager can manage agents with identical agent names."""
    with patch("rasa.agents.agent_manager.AgentFactory") as mock_factory:
        mock_client = AsyncMock()
        mock_client.connect = AsyncMock()
        mock_factory.create_client.return_value = mock_client

        await agent_manager.connect_agent(
            "some_agent", ProtocolType.MCP_TASK, mock_agent_config
        )
        await agent_manager.connect_agent(
            "some_agent", ProtocolType.MCP_OPEN, mock_agent_config
        )
        await agent_manager.connect_agent(
            "some_agent", ProtocolType.A2A, mock_agent_config
        )
        await agent_manager.connect_agent(
            "some_agent", ProtocolType.MCP_TASK, mock_agent_config
        )

        # Verify all agents are stored
        assert len(agent_manager.agents) == 3
        assert agent_manager.get_agent("some_agent", ProtocolType.MCP_TASK) is not None
        assert agent_manager.get_agent("some_agent", ProtocolType.MCP_OPEN) is not None
        assert agent_manager.get_agent("some_agent", ProtocolType.A2A) is not None


def test_agent_instance_creation(agent_manager: AgentManager) -> None:
    """Test that AgentManager creates an instance of AgentProtocol."""
    assert agent_manager is not None
    assert agent_manager.agents == {}
