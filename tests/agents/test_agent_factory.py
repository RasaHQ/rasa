"""Unit tests for AgentFactory."""

from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

from rasa.agents.agent_factory import AgentFactory
from rasa.agents.core.agent_protocol import AgentProtocol
from rasa.agents.core.types import ProtocolType
from rasa.agents.protocol import A2AAgent, MCPOpenAgent, MCPTaskAgent
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentConnections,
    AgentInfo,
    AgentMCPServerConfig,
    ProtocolConfig,
)
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR


class TestAgentFactory:
    """Test cases for AgentFactory."""

    @pytest.fixture
    def mock_agent_info(self) -> AgentInfo:
        """Fixture for creating a mock AgentInfo."""
        return AgentInfo(
            name="test_agent", protocol=ProtocolConfig.RASA, description="A test agent"
        )

    @pytest.fixture
    def mock_agent_config(self, mock_agent_info: AgentInfo) -> AgentConfig:
        """Fixture for creating a mock AgentConfig."""
        return AgentConfig(
            agent=mock_agent_info,
            configuration=AgentConfiguration(
                llm={"provider": "openai", "model": "gpt-4"},
                prompt_template="Test template",
                timeout=30,
                max_retries=3,
            ),
            connections=AgentConnections(
                mcp_servers=[
                    AgentMCPServerConfig(
                        name="test_server",
                        command="python",
                        args=["-m", "test_server"],
                        env={},
                    )
                ]
            ),
        )

    @pytest.fixture
    def mock_custom_agent_config(self, mock_agent_info: AgentInfo) -> AgentConfig:
        """Fixture for creating a mock AgentConfig with custom module."""
        return AgentConfig(
            agent=mock_agent_info,
            configuration=AgentConfiguration(
                module="test.custom_agent.CustomAgent",
                llm={"provider": "openai", "model": "gpt-4"},
            ),
        )

    def test_protocols_mapping(self) -> None:
        """Test that the protocols mapping is correct."""
        expected_mapping = {
            ProtocolType.A2A: A2AAgent,
            ProtocolType.MCP_OPEN: MCPOpenAgent,
            ProtocolType.MCP_TASK: MCPTaskAgent,
        }

        assert AgentFactory._protocols == expected_mapping

    def test_get_supported_protocols(self) -> None:
        """Test getting supported protocol types."""
        result = AgentFactory.get_supported_protocols()

        expected = [
            ProtocolType.A2A,
            ProtocolType.MCP_OPEN,
            ProtocolType.MCP_TASK,
        ]
        assert set(result) == set(expected)
        assert len(result) == 3

    def test_is_protocol_supported_true(self) -> None:
        """Test checking if supported protocol is supported."""
        assert AgentFactory.is_protocol_supported(ProtocolType.A2A) is True
        assert AgentFactory.is_protocol_supported(ProtocolType.MCP_OPEN) is True
        assert AgentFactory.is_protocol_supported(ProtocolType.MCP_TASK) is True

    def test_is_protocol_supported_false(self) -> None:
        """Test checking if unsupported protocol is not supported."""
        # Create a mock protocol type that's not in the mapping
        mock_protocol = MagicMock()
        mock_protocol.value = "unsupported_protocol"
        assert AgentFactory.is_protocol_supported(mock_protocol) is False

    def test_get_agent_class_from_protocol_supported(self) -> None:
        """Test getting agent class for supported protocol types."""
        a2a_class = AgentFactory._get_agent_class_from_protocol(ProtocolType.A2A)
        mcp_open_class = AgentFactory._get_agent_class_from_protocol(
            ProtocolType.MCP_OPEN
        )
        mcp_task_class = AgentFactory._get_agent_class_from_protocol(
            ProtocolType.MCP_TASK
        )

        assert a2a_class is A2AAgent
        assert mcp_open_class is MCPOpenAgent
        assert mcp_task_class is MCPTaskAgent

    def test_get_agent_class_from_protocol_unsupported(self) -> None:
        """Test getting agent class for unsupported protocol types."""
        mock_protocol = MagicMock()
        mock_protocol.value = "unsupported_protocol"

        with pytest.raises(ValueError, match="Unsupported protocol"):
            AgentFactory._get_agent_class_from_protocol(mock_protocol)

    def test_register_protocol_new(self) -> None:
        """Test registering a new protocol."""
        # Create a mock protocol type and class
        mock_protocol = MagicMock()
        mock_protocol.value = "new_protocol"
        mock_agent_class = MagicMock(spec=AgentProtocol)

        # Register the new protocol
        AgentFactory.register_protocol(mock_protocol, mock_agent_class)

        # Verify it's now supported
        assert AgentFactory.is_protocol_supported(mock_protocol) is True
        assert (
            AgentFactory._get_agent_class_from_protocol(mock_protocol)
            is mock_agent_class
        )

        # Clean up - remove the registered protocol
        del AgentFactory._protocols[mock_protocol]

    def test_register_protocol_already_exists(self) -> None:
        """Test registering a protocol that already exists."""
        with pytest.raises(ValueError, match="Protocol A2A already registered"):
            AgentFactory.register_protocol(ProtocolType.A2A, A2AAgent)

    def test_is_valid_custom_agent_valid(self) -> None:
        """Test checking if a valid custom agent class is valid."""

        # Create a mock custom agent class that subclasses MCPOpenAgent
        class CustomMCPAgent(MCPOpenAgent):
            pass

        result = AgentFactory._is_valid_custom_agent(
            CustomMCPAgent, ProtocolType.MCP_OPEN
        )
        assert result is True

    def test_is_valid_custom_agent_invalid(self) -> None:
        """Test checking if an invalid custom agent class is invalid."""

        # Create a mock class that doesn't subclass the expected protocol class
        class InvalidAgent:
            pass

        result = AgentFactory._is_valid_custom_agent(
            InvalidAgent, ProtocolType.MCP_OPEN
        )
        assert result is False

    @patch("rasa.agents.agent_factory.class_from_module_path")
    def test_create_client_custom_agent_valid(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_agent_config: AgentConfig,
    ) -> None:
        """Test creating a client with a valid custom agent."""

        # Create a mock custom agent class
        class CustomMCPAgent(MCPOpenAgent):
            pass

        mock_class_from_module_path.return_value = CustomMCPAgent

        # Mock the from_config method
        with patch.object(CustomMCPAgent, "from_config") as mock_from_config:
            mock_agent_instance = MagicMock(spec=AgentProtocol)
            mock_from_config.return_value = mock_agent_instance

            result = AgentFactory.create_client(
                ProtocolType.MCP_OPEN, mock_custom_agent_config
            )

            # Verify the custom agent was created
            mock_class_from_module_path.assert_called_once_with(
                "test.custom_agent.CustomAgent"
            )
            mock_from_config.assert_called_once_with(mock_custom_agent_config)
            assert result is mock_agent_instance

    @patch("rasa.agents.agent_factory.class_from_module_path")
    def test_create_client_custom_agent_invalid(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_agent_config: AgentConfig,
    ) -> None:
        """Test creating a client with an invalid custom agent."""

        # Create a mock class that doesn't subclass the expected protocol class
        class InvalidAgent:
            pass

        mock_class_from_module_path.return_value = InvalidAgent

        with pytest.raises(
            ValueError, match="Agent class .*InvalidAgent.* does not subclass"
        ):
            AgentFactory.create_client(ProtocolType.MCP_OPEN, mock_custom_agent_config)

    @patch("rasa.agents.agent_factory.class_from_module_path")
    def test_create_client_custom_agent_module_not_found(
        self,
        mock_class_from_module_path: MagicMock,
        mock_custom_agent_config: AgentConfig,
    ) -> None:
        """Test creating a client when custom agent module is not found."""
        mock_class_from_module_path.side_effect = ImportError("Module not found")

        with pytest.raises(ImportError, match="Module not found"):
            AgentFactory.create_client(ProtocolType.MCP_OPEN, mock_custom_agent_config)

    def test_create_client_builtin_agent_a2a(
        self, mock_agent_config: AgentConfig
    ) -> None:
        """Test creating a client with built-in A2A agent."""
        # Update the config to use A2A protocol
        mock_agent_config.agent.protocol = ProtocolConfig.A2A
        mock_agent_config.configuration.agent_card = "test_agent_card.json"

        # Create the actual agent instance
        result = AgentFactory.create_client(ProtocolType.A2A, mock_agent_config)

        # Verify it's an A2A agent instance
        assert isinstance(result, A2AAgent)
        assert result._name == "test_agent"
        assert result._description == "A test agent"
        assert result.protocol_type == ProtocolType.A2A

    def test_create_client_builtin_agent_mcp_open(
        self,
        mock_agent_config: AgentConfig,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test creating a client with built-in MCP Open agent."""
        monkeypatch.setenv(
            OPENAI_API_KEY_ENV_VAR,
            "mock key in test_create_client_builtin_agent_mcp_open",
        )

        # Create the actual agent instance
        result = AgentFactory.create_client(ProtocolType.MCP_OPEN, mock_agent_config)

        # Verify it's an MCP Open agent instance
        assert isinstance(result, MCPOpenAgent)
        assert result._name == "test_agent"
        assert result._description == "A test agent"
        assert result.protocol_type == ProtocolType.MCP_OPEN

    def test_create_client_builtin_agent_mcp_task(
        self,
        mock_agent_config: AgentConfig,
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test creating a client with built-in MCP Task agent."""
        monkeypatch.setenv(
            OPENAI_API_KEY_ENV_VAR,
            "mock key in test_create_client_builtin_agent_mcp_task",
        )

        # Create the actual agent instance
        result = AgentFactory.create_client(ProtocolType.MCP_TASK, mock_agent_config)

        # Verify it's an MCP Task agent instance
        assert isinstance(result, MCPTaskAgent)
        assert result._name == "test_agent"
        assert result._description == "A test agent"
        assert result.protocol_type == ProtocolType.MCP_TASK

    def test_create_client_unsupported_protocol(
        self, mock_agent_config: AgentConfig
    ) -> None:
        """Test creating a client with unsupported protocol."""
        mock_protocol = MagicMock()
        mock_protocol.value = "unsupported_protocol"

        with pytest.raises(ValueError, match="Unsupported protocol"):
            AgentFactory.create_client(mock_protocol, mock_agent_config)
