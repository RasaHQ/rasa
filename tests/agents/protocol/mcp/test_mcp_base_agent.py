"""Unit tests for MCPBaseAgent."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import anyio
import pytest
from pytest import MonkeyPatch

from rasa.agents.constants import (
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_SENDER_ID_KEY,
)
from rasa.agents.core.types import AgentStatus
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentOutput, AgentToolResult
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentConnections,
    AgentInfo,
    AgentMCPServerConfig,
    ProtocolConfig,
)
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_REACT_SUB_AGENT_NAME,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)

from .test_utils import MockMCPBaseAgentImpl


class TestMCPBaseAgent:
    """Test cases for MCPBaseAgent."""

    @pytest.fixture
    def mock_agent_input(self) -> AgentInput:
        """Fixture for creating a mock AgentInput."""
        return AgentInput(
            id="test_id",
            user_message="Hello, how can you help me?",
            slots=[
                AgentInputSlot(
                    name="user_name", value="John", type="text", allowed_values=None
                ),
                AgentInputSlot(
                    name="user_age", value=25, type="number", allowed_values=None
                ),
            ],
            conversation_history="Previous conversation...",
            events=[],
            metadata={"key": "value", "nested": {"data": "test"}},
            timestamp="2024-01-15T10:30:00Z",
        )

    @pytest.fixture
    def mock_agent_config(self) -> AgentConfig:
        """Fixture for creating a mock AgentConfig."""
        return AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="A test agent for unit testing",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                llm={"provider": "openai", "model": "gpt-4"},
                prompt_template="Test template: {{user_message}}",
                timeout=30,
                max_retries=3,
            ),
            connections=AgentConnections(
                mcp_servers=[
                    AgentMCPServerConfig(
                        name="test_server",
                        url="http://localhost:8000",
                        include_tools=["tool1", "tool2"],
                        exclude_tools=["tool3"],
                    )
                ]
            ),
        )

    @pytest.fixture
    def mock_mcp_base_agent(self, monkeypatch: MonkeyPatch) -> MockMCPBaseAgentImpl:
        """Fixture for creating a mock MCPBaseAgent instance."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_mcp_base_agent")

        return MockMCPBaseAgentImpl(
            name="test_agent",
            description="A test agent",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
        )

    # ============================================================================
    # Initialization & Setup Tests
    # ============================================================================

    def test_init_basic_initialization(self, monkeypatch: MonkeyPatch) -> None:
        """Test basic initialization of MCPBaseAgent."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl(
            name="test_agent",
            description="Test description",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
        )

        assert agent._name == "test_agent"
        assert agent._description == "Test description"
        assert agent._protocol_type == ProtocolConfig.RASA
        assert agent._server_configs == []
        assert agent._mcp_tools == []
        assert agent._custom_tools == []
        assert agent._tool_to_server_mapper == {}
        assert agent._server_connections == {}

    def test_init_with_llm_config(self, monkeypatch: MonkeyPatch) -> None:
        """Test initialization with custom LLM config."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        llm_config = {"provider": "openai", "model": "gpt-4", "temperature": 0.7}
        agent = MockMCPBaseAgentImpl(
            "test-agent", "test", ProtocolConfig.RASA, [], llm_config=llm_config
        )

        assert agent._llm_config is not None
        assert agent.llm_client is not None

    def test_from_config(
        self, mock_agent_config: AgentConfig, monkeypatch: MonkeyPatch
    ) -> None:
        """Test from_config class method."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl.from_config(mock_agent_config)

        assert agent._name == "test_agent"
        assert agent._description == "A test agent for unit testing"
        assert agent._protocol_type == ProtocolConfig.RASA
        assert len(agent._server_configs) == 1
        assert agent._server_configs[0].name == "test_server"

    def test_from_config_timeout_warning(
        self, monkeypatch: MonkeyPatch, capsys
    ) -> None:
        """Test that from_config warns when configuration.timeout is set."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        # Create agent config with timeout
        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="Test agent",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                llm={"provider": "openai", "model": "gpt-4"},
                timeout=30,
            ),
            connections=AgentConnections(),
        )

        # Create agent
        agent = MockMCPBaseAgentImpl.from_config(agent_config)

        # Capture output
        captured = capsys.readouterr()

        # Verify warning was logged
        expected_warning = (
            "`configuration.timeout` is not supported for MCP agents. MCP "
            "agents do not make external connections, so an agent-level timeout"
            " does not apply. To control timeout behavior for LLM calls, set "
            "the `timeout` value in the `model_group` section of endpoints.yml "
            "and reference it through `configuration.llm.model_group`."
        )
        assert "mcp_agent.configuration.timeout.not_implemented" in captured.out
        assert expected_warning in captured.out

        # Verify agent was still created successfully
        assert agent._name == "test_agent"

    # ============================================================================
    # Class Configuration & Properties Tests
    # ============================================================================

    def test_agent_conforms_to(self, mock_mcp_base_agent: MockMCPBaseAgentImpl) -> None:
        """Test agent_conforms_to property."""
        assert mock_mcp_base_agent.agent_conforms_to == ProtocolConfig.RASA

    def test_get_default_llm_config(self) -> None:
        """Test get_default_llm_config static method."""
        config = MCPBaseAgent.get_default_llm_config()

        assert config["provider"] == "openai"
        assert config["model"] == "gpt-4o-2024-11-20"
        assert config["temperature"] == 0.0
        assert config["max_completion_tokens"] == 256
        assert config["timeout"] == 7

    def test_get_agent_specific_built_in_tools(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test get_agent_specific_built_in_tools method."""
        tools = mock_mcp_base_agent.get_agent_specific_built_in_tools(mock_agent_input)

        assert isinstance(tools, list)
        assert len(tools) == 0  # Default implementation returns empty list

    def test_get_custom_tool_definitions(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test get_custom_tool_definitions method."""
        tools = mock_mcp_base_agent.get_custom_tool_definitions()

        assert isinstance(tools, list)
        assert len(tools) == 0  # Default implementation returns empty list

    # ============================================================================
    # Connection Management Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_connect_success(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test successful connection to MCP servers."""
        with (
            patch.object(
                mock_mcp_base_agent, "connect_to_servers"
            ) as mock_connect_servers,
            patch.object(
                mock_mcp_base_agent, "fetch_and_store_available_tools"
            ) as mock_fetch_tools,
        ):
            await mock_mcp_base_agent.connect()

            mock_connect_servers.assert_called_once()
            mock_fetch_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_connection_error_with_retries(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test connection with retries on ConnectionError."""
        with patch.object(
            mock_mcp_base_agent, "connect_to_servers"
        ) as mock_connect_servers:
            mock_connect_servers.side_effect = ConnectionError("Connection failed")

            with pytest.raises(
                Exception
            ):  # Should raise AgentInitializationException after retries
                await mock_mcp_base_agent.connect()

    @pytest.mark.asyncio
    async def test_connect_to_server_success(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test successful connection to a single server."""
        server_config = AgentMCPServerConfig(
            name="test_server",
            url="http://localhost:8000",
        )

        mock_connection = MagicMock()
        mock_connection.connect = AsyncMock()
        mock_connection.server_url = "http://localhost:8000"

        with patch(
            "rasa.shared.utils.mcp.server_connection.MCPServerConnection.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = mock_connection

            await mock_mcp_base_agent.connect_to_server(server_config)

            mock_connection.connect.assert_called_once()
            assert "test_server" in mock_mcp_base_agent._server_connections

    @pytest.mark.asyncio
    async def test_connect_to_server_failure(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test connection failure to a single server."""
        server_config = AgentMCPServerConfig(
            name="test_server",
            url="http://localhost:8000",
        )

        mock_connection = MagicMock()
        mock_connection.connect = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "rasa.shared.utils.mcp.server_connection.MCPServerConnection.from_config"
        ) as mock_from_config:
            mock_from_config.return_value = mock_connection

            with pytest.raises(Exception):
                await mock_mcp_base_agent.connect_to_server(server_config)

    @pytest.mark.asyncio
    async def test_disconnect_server_success(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test successful disconnection from a server."""
        mock_connection = MagicMock()
        mock_connection.close = AsyncMock()
        mock_mcp_base_agent._server_connections["test_server"] = mock_connection

        await mock_mcp_base_agent.disconnect_server("test_server")

        mock_connection.close.assert_called_once()
        assert "test_server" in mock_mcp_base_agent._server_connections

    @pytest.mark.asyncio
    async def test_disconnect_server_not_found(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test disconnection from a non-existent server."""
        # Should not raise an exception
        await mock_mcp_base_agent.disconnect_server("non_existent_server")

    @pytest.mark.asyncio
    async def test_disconnect_all_servers(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test disconnection from all servers."""
        mock_connection1 = MagicMock()
        mock_connection1.close = AsyncMock()
        mock_connection2 = MagicMock()
        mock_connection2.close = AsyncMock()

        mock_mcp_base_agent._server_connections = {
            "server1": mock_connection1,
            "server2": mock_connection2,
        }

        await mock_mcp_base_agent.disconnect()

        mock_connection1.close.assert_called_once()
        mock_connection2.close.assert_called_once()
        assert len(mock_mcp_base_agent._server_connections) == 2

    # ============================================================================
    # Tool Management Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_list_tools(self, mock_mcp_base_agent: MockMCPBaseAgentImpl) -> None:
        """Test listing tools from MCP server."""
        mock_connection = MagicMock()
        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock())
        mock_connection.ensure_active_session = AsyncMock(return_value=mock_session)

        await mock_mcp_base_agent.list_tools(mock_connection)

        mock_connection.ensure_active_session.assert_called_once()
        mock_session.list_tools.assert_called_once()

    def test_get_custom_tools(self, mock_mcp_base_agent: MockMCPBaseAgentImpl) -> None:
        """Test getting custom tools."""
        from rasa.agents.schemas import AgentToolSchema

        # Add some custom tools using AgentToolSchema
        tool1 = AgentToolSchema(
            name="tool1",
            description="Tool 1 description",
            parameters={"type": "object", "properties": {}},
            strict=False,
            type="function",
        )
        tool2 = AgentToolSchema(
            name="tool2",
            description="Tool 2 description",
            parameters={"type": "object", "properties": {}},
            strict=False,
            type="function",
        )

        # Create mock objects with tool_definition attribute
        mock_tool1 = MagicMock()
        mock_tool1.tool_definition = tool1
        mock_tool2 = MagicMock()
        mock_tool2.tool_definition = tool2

        mock_mcp_base_agent._custom_tools = [mock_tool1, mock_tool2]

        tools = mock_mcp_base_agent.get_custom_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool1"
        assert tools[1].name == "tool2"

    def test_get_available_tools(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test getting all available tools."""
        # Add some MCP tools
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.name = "mcp_tool"
        mock_mcp_base_agent._mcp_tools = [mock_mcp_tool]

        # Add some custom tools
        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_definition = {"name": "custom_tool"}
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        with patch.object(
            mock_mcp_base_agent, "get_agent_specific_built_in_tools"
        ) as mock_built_in:
            mock_built_in.return_value = []

            tools = mock_mcp_base_agent.get_available_tools(mock_agent_input)

            assert len(tools) == 2
            assert tools[0].name == "mcp_tool"
            assert tools[1]["name"] == "custom_tool"

    @pytest.mark.asyncio
    async def test_get_filtered_tools_from_server_include_tools(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting filtered tools with include_tools filter."""
        mock_connection = MagicMock()
        mock_tools_response = MagicMock()

        # Create proper mock tools with all required attributes
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1 description"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2 description"
        mock_tool2.inputSchema = {"type": "object", "properties": {}}

        mock_tool3 = MagicMock()
        mock_tool3.name = "tool3"
        mock_tool3.description = "Tool 3 description"
        mock_tool3.inputSchema = {"type": "object", "properties": {}}

        mock_tools_response.tools = [mock_tool1, mock_tool2, mock_tool3]

        with patch.object(mock_mcp_base_agent, "list_tools") as mock_list_tools:
            mock_list_tools.return_value = mock_tools_response

            tools = await mock_mcp_base_agent._get_filtered_tools_from_server(
                "test_server", mock_connection, include_tools=["tool1", "tool3"]
            )

            assert len(tools) == 2
            assert tools[0].name == "tool1"
            assert tools[1].name == "tool3"

    @pytest.mark.asyncio
    async def test_get_filtered_tools_from_server_exclude_tools(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting filtered tools with exclude_tools filter."""
        mock_connection = MagicMock()
        mock_tools_response = MagicMock()

        # Create proper mock tools with all required attributes
        mock_tool1 = MagicMock()
        mock_tool1.name = "tool1"
        mock_tool1.description = "Tool 1 description"
        mock_tool1.inputSchema = {"type": "object", "properties": {}}

        mock_tool2 = MagicMock()
        mock_tool2.name = "tool2"
        mock_tool2.description = "Tool 2 description"
        mock_tool2.inputSchema = {"type": "object", "properties": {}}

        mock_tool3 = MagicMock()
        mock_tool3.name = "tool3"
        mock_tool3.description = "Tool 3 description"
        mock_tool3.inputSchema = {"type": "object", "properties": {}}

        mock_tools_response.tools = [mock_tool1, mock_tool2, mock_tool3]

        with patch.object(mock_mcp_base_agent, "list_tools") as mock_list_tools:
            mock_list_tools.return_value = mock_tools_response

            tools = await mock_mcp_base_agent._get_filtered_tools_from_server(
                "test_server", mock_connection, exclude_tools=["tool2"]
            )

            assert len(tools) == 2
            assert tools[0].name == "tool1"
            assert tools[1].name == "tool3"

    def test_get_include_exclude_tools_from_server_configs(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting include/exclude tools from server configs."""
        server_config = AgentMCPServerConfig(
            name="test_server",
            url="http://localhost:8000",
            include_tools=["tool1", "tool2"],
            exclude_tools=["tool3"],
        )
        mock_mcp_base_agent._server_configs = [server_config]

        include_tools, exclude_tools = (
            mock_mcp_base_agent._get_include_exclude_tools_from_server_configs(
                "test_server"
            )
        )

        assert include_tools == ["tool1", "tool2"]
        assert exclude_tools == ["tool3"]

    def test_get_include_exclude_tools_from_server_configs_not_found(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting include/exclude tools for non-existent server."""
        include_tools, exclude_tools = (
            mock_mcp_base_agent._get_include_exclude_tools_from_server_configs(
                "non_existent"
            )
        )

        assert include_tools is None
        assert exclude_tools is None

    # ============================================================================
    # LLM & Prompt Management Tests
    # ============================================================================

    def test_render_prompt_template(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test rendering prompt template with context."""

        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC"))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            result = mock_mcp_base_agent.render_prompt_template(mock_agent_input)

            assert "Hello, how can you help me?" in result
            assert "Previous conversation..." in result
            assert "- Current date: 15 January, 2024" in result
            assert "- Current time: 14:30:45 (UTC)" in result
            assert "- Current day: Monday" in result

    @pytest.mark.parametrize(
        "slots, expected_assertions",
        [
            # Test basic slot access
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=25,
                        type="float",
                    ),
                ],
                [
                    ("user_name=John", True),
                    ("user_age=25", True),
                    ("Direct: John, 25", True),
                ],
            ),
            # Test None values are excluded
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=None,
                        type="float",
                    ),
                    AgentInputSlot(
                        name="user_email",
                        value="john@example.com",
                        type="text",
                    ),
                ],
                [
                    ("user_name=John", True),
                    ("user_email=john@example.com", True),
                    ("user_age=None", False),  # Should not appear
                    ("Has user_age: no", True),  # Should not be in dict
                ],
            ),
            # Test empty slots
            (
                [],
                [
                    ("Slots count: 0", True),
                ],
            ),
            # Test dict structure and membership check
            (
                [
                    AgentInputSlot(
                        name="user_name",
                        value="John",
                        type="text",
                    ),
                    AgentInputSlot(
                        name="user_age",
                        value=None,
                        type="float",
                    ),
                ],
                [
                    ("Direct: John", True),
                    ("Has user_name: yes", True),
                    ("Has user_age: no", True),  # None value excluded
                ],
            ),
        ],
    )
    def test_render_prompt_template_slots_access(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        slots: List[AgentInputSlot],
        expected_assertions: List[Tuple[str, bool]],
    ) -> None:
        """Test that slots are accessible in the prompt template as a dict."""
        # Single comprehensive template that covers all test cases
        template = (
            "User message: {{user_message}}\n"
            "Slots count: {{ slots|length }}\n"
            "Slots: {% for slot_name, slot_value in slots.items() %}"
            "{{ slot_name }}={{ slot_value }}"
            "{% endfor %}\n"
            "Direct: {{ slots.user_name if slots.user_name else 'not set' }}, "
            "{{ slots.user_age if slots.user_age else 'not set' }}\n"
            "Has user_name: {{ 'yes' if 'user_name' in slots else 'no' }}\n"
            "Has user_age: {{ 'yes' if 'user_age' in slots else 'no' }}"
        )
        mock_mcp_base_agent.prompt_template = template

        agent_input = AgentInput(
            id="test_id",
            user_message="Hello",
            slots=slots,
            conversation_history="",
            events=[],
            metadata={},
            timestamp="2024-01-15T10:30:00Z",
        )

        result = mock_mcp_base_agent.render_prompt_template(agent_input)

        # Verify all expected assertions
        for expected_text, should_be_present in expected_assertions:
            if should_be_present:
                assert expected_text in result, f"Expected '{expected_text}' in result"
            else:
                assert (
                    expected_text not in result
                ), f"Expected '{expected_text}' NOT in result"

    def test_build_messages_for_llm_request(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test building messages for LLM request."""
        # Add some events to the input
        mock_agent_input.events = [
            UserUttered(text="Hello"),
            BotUttered(text="Hi there"),
            UserUttered(text="How are you?"),
        ]

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"

            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

            assert len(messages) >= 1  # At least system message
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "System prompt"

    @pytest.mark.parametrize(
        "tool_calls, expected_result_keys",
        [
            # With tool calls
            (
                [
                    LLMToolCall(
                        id="call_123",
                        type="function",
                        tool_name="test_tool",
                        tool_args={"arg1": "value1"},
                    )
                ],
                ["role", "content", "tool_calls"],
            ),
            # Without tool calls
            ([], []),
        ],
    )
    def test_get_assistant_message_with_tool_calls(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        tool_calls,
        expected_result_keys,
    ) -> None:
        """Test getting assistant message with and without tool calls."""
        llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=tool_calls,
        )

        result = mock_mcp_base_agent._get_assistant_message_with_tool_calls(
            llm_response
        )

        if expected_result_keys:
            assert result["role"] == "assistant"
            assert result["content"] == "Test response"
            if "tool_calls" in expected_result_keys:
                assert len(result["tool_calls"]) == 1
                assert result["tool_calls"][0]["id"] == "call_123"
                assert result["tool_calls"][0]["function"]["name"] == "test_tool"
        else:
            assert result == {}

    def test_get_tool_call_message(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting tool call message."""
        tool_response = AgentOutput(
            id="call_123",
            status=AgentStatus.COMPLETED,
            response_message="Tool result",
        )

        result = mock_mcp_base_agent._get_tool_call_message(tool_response)

        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "Tool result"

    def test_get_system_message_for_malformed_tool_response(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test getting system message for malformed tool response."""
        result = mock_mcp_base_agent._get_system_message_for_malformed_tool_response()

        assert result["role"] == "system"
        assert "invalid" in result["content"].lower()
        assert "JSON" in result["content"]

    # ============================================================================
    # Tool Execution Tests
    # ============================================================================

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name, setup_connection, expected_success, expected_error_keyword",
        [
            # Success case
            (
                "test_tool",
                True,
                True,
                None,
            ),
            # Tool not found
            (
                "non_existent_tool",
                False,
                False,
                "not found",
            ),
            # Exception case
            (
                "test_tool",
                "exception",
                False,
                "Failed to execute tool",
            ),
        ],
    )
    async def test_execute_mcp_tool_scenarios(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        tool_name,
        setup_connection,
        expected_success,
        expected_error_keyword,
    ) -> None:
        """Test MCP tool execution in various scenarios."""
        if setup_connection is True:
            # Setup successful connection
            mock_connection = MagicMock()
            mock_session = MagicMock()
            mock_session.call_tool = AsyncMock(return_value={"result": "success"})
            mock_connection.ensure_active_session = AsyncMock(return_value=mock_session)
            mock_connection.server_url = "http://localhost:8000"

            mock_mcp_base_agent._tool_to_server_mapper["test_tool"] = "test_server"
            mock_mcp_base_agent._server_connections["test_server"] = mock_connection

            with patch(
                "rasa.agents.schemas.AgentToolResult.from_mcp_tool_result"
            ) as mock_from_mcp:
                mock_from_mcp.return_value = AgentToolResult(
                    tool_name="test_tool",
                    result='{"result": "success"}',
                    is_error=False,
                )

                result = await mock_mcp_base_agent._execute_mcp_tool(
                    tool_name, {"arg": "value"}
                )

                if expected_success:
                    mock_session.call_tool.assert_called_once_with(
                        tool_name,
                        {"arg": "value"},
                        read_timeout_seconds=timedelta(seconds=10),
                    )
                    assert result.tool_name == tool_name
                    assert not result.is_error
        elif setup_connection == "exception":
            # Setup connection with exception
            mock_connection = MagicMock()
            mock_connection.ensure_active_session = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_connection.server_url = "http://localhost:8000"

            mock_mcp_base_agent._tool_to_server_mapper["test_tool"] = "test_server"
            mock_mcp_base_agent._server_connections["test_server"] = mock_connection

            result = await mock_mcp_base_agent._execute_mcp_tool(tool_name, {})

            assert result.tool_name == tool_name
            assert result.is_error
            assert expected_error_keyword in result.error_message
        else:
            # Tool not found case
            result = await mock_mcp_base_agent._execute_mcp_tool(tool_name, {})

            assert result.tool_name == tool_name
            assert result.is_error
            assert expected_error_keyword in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_call_custom_tool(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test executing custom tool call."""
        mock_tool_executor = AsyncMock(
            return_value=AgentToolResult(
                tool_name="custom_tool",
                result="custom_result",
                is_error=False,
            )
        )

        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "custom_tool"
        mock_custom_tool.tool_executor = mock_tool_executor
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        result = await mock_mcp_base_agent._execute_tool_call(
            "custom_tool", {"arg": "value"}
        )

        mock_tool_executor.assert_called_once_with({"arg": "value"})
        assert result.tool_name == "custom_tool"
        assert not result.is_error

    @pytest.mark.asyncio
    async def test_execute_tool_call_custom_tool_exception(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test executing custom tool call with exception."""
        mock_tool_executor = MagicMock(side_effect=Exception("Custom tool failed"))

        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "custom_tool"
        mock_custom_tool.tool_executor = mock_tool_executor
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        result = await mock_mcp_base_agent._execute_tool_call("custom_tool", {})

        assert result.tool_name == "custom_tool"
        assert result.is_error
        assert "Failed to execute built-in tool" in result.error_message

    @pytest.mark.asyncio
    async def test_execute_tool_call_mcp_tool(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test executing MCP tool call."""
        with patch.object(mock_mcp_base_agent, "_execute_mcp_tool") as mock_execute_mcp:
            mock_execute_mcp.return_value = AgentToolResult(
                tool_name="mcp_tool",
                result="mcp_result",
                is_error=False,
            )

            result = await mock_mcp_base_agent._execute_tool_call(
                "mcp_tool", {"arg": "value"}
            )

            mock_execute_mcp.assert_called_once_with("mcp_tool", {"arg": "value"})
            assert result.tool_name == "mcp_tool"

    @pytest.mark.parametrize(
        "is_error, error_message, expected_status",
        [
            (True, "Fatal error", "FATAL_ERROR"),
            (False, "Recoverable error", "RECOVERABLE_ERROR"),
        ],
    )
    def test_generate_agent_error_output(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
        is_error,
        error_message,
        expected_status,
    ) -> None:
        """Test generating agent error output for different error types."""
        tool_output = AgentToolResult(
            tool_name="test_tool",
            result=None,
            is_error=is_error,
            error_message=error_message,
        )

        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="test_tool",
            tool_args={"arg": "value"},
        )

        result = mock_mcp_base_agent._generate_agent_error_output(
            tool_output, mock_agent_input, tool_call
        )

        assert result.id == mock_agent_input.id
        assert result.status.name == expected_status
        assert result.error_message == error_message

    def test_get_structured_results_for_agent_output(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test getting structured results for agent output."""
        tool_results = {
            "tool1": AgentToolResult(
                tool_name="tool1",
                result='{"data": "result1"}',
                is_error=False,
            ),
            "tool2": AgentToolResult(
                tool_name="tool2",
                result='{"data": "result2"}',
                is_error=False,
            ),
        }

        result = mock_mcp_base_agent._get_structured_results_for_agent_output(
            mock_agent_input, tool_results
        )

        assert len(result) == 1  # One iteration
        assert len(result[0]) == 2  # Two tools
        assert result[0][0]["name"] == "tool1"
        assert result[0][0]["result"] == '{"data": "result1"}'
        assert result[0][1]["name"] == "tool2"
        assert result[0][1]["result"] == '{"data": "result2"}'

    # ============================================================================
    # Core Protocol Methods Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_run_calls_send_message(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test that run method calls send_message."""
        with patch.object(mock_mcp_base_agent, "send_message") as mock_send_message:
            mock_send_message.return_value = AgentOutput(
                id="test_id",
                status=AgentStatus.COMPLETED,
            )

            await mock_mcp_base_agent.run(mock_agent_input)

            mock_send_message.assert_called_once_with(mock_agent_input, None)

    # ============================================================================
    # Message Processing Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_process_input_returns_same_input(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test that process_input returns the same input."""
        result = await mock_mcp_base_agent.process_input(mock_agent_input)

        assert result == mock_agent_input

    @pytest.mark.asyncio
    async def test_process_output_returns_same_output(self, mock_mcp_base_agent):
        """Test that process_output returns the same output."""
        output = AgentOutput(
            id="test_id",
            status=AgentStatus.COMPLETED,
        )

        result = await mock_mcp_base_agent.process_output(output)

        assert result == output

    # ============================================================================
    # Timeout Tests
    # ============================================================================

    @pytest.mark.asyncio
    async def test_custom_tool_timeout_with_anyio_fail_after(self, mock_mcp_base_agent):
        """Test that custom tool times out using anyio.fail_after()."""
        # Setup custom tool that will timeout
        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "timeout_tool"

        # Set short timeout for test
        mock_mcp_base_agent.TOOL_CALL_DEFAULT_TIMEOUT = 0.1

        # Mock tool executor that sleeps longer than timeout
        async def slow_tool_executor(args):
            await anyio.sleep(0.2)  # Sleep longer than TOOL_CALL_DEFAULT_TIMEOUT
            return AgentToolResult(
                tool_name="timeout_tool",
                result="result",
                is_error=False,
                error_message=None,
            )

        mock_custom_tool.tool_executor = slow_tool_executor
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        # Execute
        result = await mock_mcp_base_agent._execute_tool_call(
            "timeout_tool", {"arg": "value"}
        )

        # Assert timeout error
        assert result.tool_name == "timeout_tool"
        assert result.is_error is True
        assert "timed out after" in result.error_message
        assert "seconds" in result.error_message
        # Assert error message format
        expected_message = (
            "Built-in tool `timeout_tool` timed out after "
            f"{mock_mcp_base_agent.TOOL_CALL_DEFAULT_TIMEOUT} seconds."
        )
        assert result.error_message == expected_message

    @pytest.mark.asyncio
    async def test_custom_tool_success_within_timeout(self, mock_mcp_base_agent):
        """Test that custom tool executes successfully within timeout."""
        # Setup custom tool that completes quickly

        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "fast_tool"

        # Set short timeout for test
        mock_mcp_base_agent.TOOL_CALL_DEFAULT_TIMEOUT = 0.1

        # Mock tool executor that completes quickly
        async def fast_tool_executor(args):
            return AgentToolResult(
                tool_name="fast_tool",
                result="success",
                is_error=False,
                error_message=None,
            )

        mock_custom_tool.tool_executor = fast_tool_executor
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        # Execute
        result = await mock_mcp_base_agent._execute_tool_call(
            "fast_tool", {"arg": "value"}
        )

        # Assert success
        assert result.tool_name == "fast_tool"
        assert result.is_error is False
        assert result.result == "success"

    @pytest.mark.asyncio
    async def test_custom_tool_exception_during_execution(self, mock_mcp_base_agent):
        """Test that exceptions during tool execution are handled properly."""
        # Setup custom tool that raises exception
        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "failing_tool"

        # Mock tool executor that raises exception
        async def failing_tool_executor(args):
            raise ValueError("Tool execution failed")

        mock_custom_tool.tool_executor = failing_tool_executor
        mock_mcp_base_agent._custom_tools = [mock_custom_tool]

        # Execute
        result = await mock_mcp_base_agent._execute_tool_call(
            "failing_tool", {"arg": "value"}
        )

        # Assert error handling
        assert result.tool_name == "failing_tool"
        assert result.is_error is True
        assert "Failed to execute built-in tool" in result.error_message
        assert "Tool execution failed" in result.error_message

    # ============================================================================
    # get_llm_tracing_metadata Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "sender_id, agent_id, model_id, expected_metadata",
        [
            (
                "user123",
                "assistant456",
                "model789",
                {
                    LANGFUSE_METADATA_SESSION_ID: "user123",
                    LANGFUSE_METADATA_TAGS: [MockMCPBaseAgentImpl.__name__],
                    LANGFUSE_METADATA_CUSTOM_METADATA: {
                        LANGFUSE_METADATA_AGENT_ID: "assistant456",
                        LANGFUSE_METADATA_MODEL_ID: "model789",
                        LANGFUSE_METADATA_COMPONENT_NAME: MockMCPBaseAgentImpl.__name__,
                        LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: "test_agent",
                    },
                },
            ),
            (
                "user123",
                None,
                None,
                {
                    LANGFUSE_METADATA_SESSION_ID: "user123",
                    LANGFUSE_METADATA_TAGS: [MockMCPBaseAgentImpl.__name__],
                    LANGFUSE_METADATA_CUSTOM_METADATA: {
                        LANGFUSE_METADATA_AGENT_ID: None,
                        LANGFUSE_METADATA_MODEL_ID: None,
                        LANGFUSE_METADATA_COMPONENT_NAME: MockMCPBaseAgentImpl.__name__,
                        LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: "test_agent",
                    },
                },
            ),
            (
                "user123",
                "assistant456",
                None,
                {
                    LANGFUSE_METADATA_SESSION_ID: "user123",
                    LANGFUSE_METADATA_TAGS: [MockMCPBaseAgentImpl.__name__],
                    LANGFUSE_METADATA_CUSTOM_METADATA: {
                        LANGFUSE_METADATA_AGENT_ID: "assistant456",
                        LANGFUSE_METADATA_MODEL_ID: None,
                        LANGFUSE_METADATA_COMPONENT_NAME: MockMCPBaseAgentImpl.__name__,
                        LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: "test_agent",
                    },
                },
            ),
            (
                "user123",
                None,
                "model789",
                {
                    LANGFUSE_METADATA_SESSION_ID: "user123",
                    LANGFUSE_METADATA_TAGS: [MockMCPBaseAgentImpl.__name__],
                    LANGFUSE_METADATA_CUSTOM_METADATA: {
                        LANGFUSE_METADATA_AGENT_ID: None,
                        LANGFUSE_METADATA_MODEL_ID: "model789",
                        LANGFUSE_METADATA_COMPONENT_NAME: MockMCPBaseAgentImpl.__name__,
                        LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: "test_agent",
                    },
                },
            ),
            (
                None,
                "assistant456",
                "model789",
                {
                    LANGFUSE_METADATA_SESSION_ID: None,
                    LANGFUSE_METADATA_TAGS: [MockMCPBaseAgentImpl.__name__],
                    LANGFUSE_METADATA_CUSTOM_METADATA: {
                        LANGFUSE_METADATA_AGENT_ID: "assistant456",
                        LANGFUSE_METADATA_MODEL_ID: "model789",
                        LANGFUSE_METADATA_COMPONENT_NAME: MockMCPBaseAgentImpl.__name__,
                        LANGFUSE_METADATA_REACT_SUB_AGENT_NAME: "test_agent",
                    },
                },
            ),
        ],
    )
    def test_get_llm_tracing_metadata(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        sender_id: Optional[str],
        agent_id: Optional[str],
        model_id: Optional[str],
        expected_metadata: Dict[str, Any],
    ) -> None:
        """Test that get_llm_tracing_metadata returns correct metadata from
        agent_input.
        """
        # Build metadata dict
        metadata = {}
        if sender_id is not None:
            metadata[AGENT_METADATA_SENDER_ID_KEY] = sender_id
        if agent_id is not None:
            metadata[AGENT_METADATA_AGENT_ID_KEY] = agent_id
        if model_id is not None:
            metadata[AGENT_METADATA_MODEL_ID_KEY] = model_id

        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            metadata=metadata,
            timestamp="2024-01-15T10:30:00Z",
        )

        result_metadata = mock_mcp_base_agent.get_llm_tracing_metadata(agent_input)

        assert result_metadata == expected_metadata

    # ============================================================================
    # _resolve_datetime Tests
    # ============================================================================

    @pytest.mark.parametrize(
        "mocked_dt, expected_tzname",
        [
            (
                "2024-01-15T10:30:00+00:00",
                "UTC",
            ),
            (
                "2024-01-15T10:30:00-05:00",
                "UTC-05:00",
            ),
        ],
    )
    def test_resolve_datetime_with_mocked_datetime_in_slots(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mocked_dt: str,
        expected_tzname: str,
    ) -> None:
        """render_prompt_template uses mocked_datetime when present in slots."""

        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[
                AgentInputSlot(
                    name=MOCKED_DATETIME_SLOT,
                    value=mocked_dt,
                    type="any",
                    allowed_values=None,
                ),
            ],
            conversation_history="Previous conversation...",
            events=[],
            metadata={},
            timestamp="2024-01-15T10:30:00Z",
        )

        # Call render_prompt_template which internally calls resolve_datetime
        rendered_prompt = mock_mcp_base_agent.render_prompt_template(agent_input)

        # Verify the mocked datetime is used in the rendered prompt
        assert "15 January, 2024" in rendered_prompt
        assert "10:30:00" in rendered_prompt
        assert "Monday" in rendered_prompt
        assert expected_tzname in rendered_prompt

    def test_resolve_datetime_without_mocked_datetime(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[
                AgentInputSlot(
                    name="other_slot",
                    value="some_value",
                    type="text",
                    allowed_values=None,
                ),
            ],
            conversation_history="Previous conversation...",
            events=[],
            metadata={},
            timestamp="2024-01-15T10:30:00Z",
        )

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current:
            expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            mock_get_current.return_value = expected_dt

            # Call render_prompt_template which internally calls resolve_datetime
            # which calls get_current_datetime when mocked_datetime is None
            rendered_prompt = mock_mcp_base_agent.render_prompt_template(agent_input)

            # Verify get_current_datetime was called through resolve_datetime
            mock_get_current.assert_called_once_with(
                timezone=mock_mcp_base_agent._timezone
            )

            # Verify the current datetime is used in the rendered prompt
            assert "15 January, 2024" in rendered_prompt
            assert "10:30:00" in rendered_prompt
            assert "Monday" in rendered_prompt

    def test_resolve_datetime_with_mocked_datetime_none(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """render_prompt_template uses current datetime when mocked_datetime is None."""
        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[
                AgentInputSlot(
                    name=MOCKED_DATETIME_SLOT,
                    value=None,
                    type="any",
                    allowed_values=None,
                ),
            ],
            conversation_history="Previous conversation...",
            events=[],
            metadata={},
            timestamp="2024-01-15T10:30:00Z",
        )

        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current:
            expected_dt = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
            mock_get_current.return_value = expected_dt

            # Call render_prompt_template which internally calls resolve_datetime
            # which calls get_current_datetime when mocked_datetime is None
            rendered_prompt = mock_mcp_base_agent.render_prompt_template(agent_input)

            # Verify get_current_datetime was called through resolve_datetime
            mock_get_current.assert_called_once_with(
                timezone=mock_mcp_base_agent._timezone
            )

            # Verify the current datetime is used in the rendered prompt
            assert "15 January, 2024" in rendered_prompt
            assert "10:30:00" in rendered_prompt
            assert "Monday" in rendered_prompt
