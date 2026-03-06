"""Unit tests for MCPBaseAgent."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import anyio
import pytest
from pytest import MonkeyPatch

from rasa.agents.constants import (
    AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE,
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_AGENT_RESPONSE_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_RESTARTED_KEY,
    AGENT_METADATA_RESUMED_AFTER_INTERRUPTION,
    AGENT_METADATA_SENDER_ID_KEY,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE,
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
from rasa.shared.core.events import (
    AgentCompleted,
    AgentStarted,
    BotUttered,
    SlotSet,
    UserUttered,
)
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
                # prompt_template is set to None to use the default template
                # (prompt_template is interpreted as a file path, not template content)
                prompt_template=None,
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
        assert config["model"] == "gpt-5.1-2025-11-13"
        assert config["reasoning_effort"] == "none"
        assert config["temperature"] == 1.0
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

    def test_render_prompt_template_includes_resume_instruction_when_resumed(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Prompt includes resume block when metadata has resumed_after_interruption."""
        last_request = "What is your budget for the car?"
        agent_input = AgentInput(
            id="test_id",
            user_message="50k",
            slots=[],
            conversation_history="...",
            events=[],
            metadata={
                AGENT_METADATA_RESUMED_AFTER_INTERRUPTION: True,
                AGENT_METADATA_AGENT_RESPONSE_KEY: last_request,
            },
        )
        result = mock_mcp_base_agent.render_prompt_template(agent_input)
        assert "Resume:" in result
        assert last_request in result

    def test_render_prompt_template_omits_resume_instruction_when_not_resumed(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """When metadata does not have resumed_after_interruption, no resume block."""
        result = mock_mcp_base_agent.render_prompt_template(mock_agent_input)
        assert "Resume:" not in result

    def test_render_prompt_template_includes_restart_instruction_when_restarted(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """When restarted=True, prompt includes restart instruction."""
        agent_input = AgentInput(
            id="my-agent",
            user_message="Run again",
            slots=[],
            conversation_history="",
            events=[],
            metadata={AGENT_METADATA_RESTARTED_KEY: True},
        )
        result = mock_mcp_base_agent.render_prompt_template(agent_input)
        assert "Agent restarted" in result
        assert "Do not set slot values from them" in result or "fresh start" in result
        assert "Conversation history" not in result

    def test_build_messages_for_llm_request_adds_markers_when_restarted(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """When restarted, messages in the completed run are prefixed with marker."""
        from rasa.agents.protocol.mcp.mcp_base_agent import PREVIOUS_RUN_MARKER

        agent_input = AgentInput(
            id="my-agent",
            user_message="New request after restart",
            slots=[],
            conversation_history="",
            events=[
                UserUttered(text="Prior user"),
                BotUttered(text="Prior bot"),
                AgentStarted(agent_id="my-agent", flow_id="test_flow"),
                UserUttered(text="User in completed run"),
                BotUttered(text="Bot in completed run"),
                AgentCompleted(agent_id="my-agent", flow_id="test_flow"),
                UserUttered(text="New request after restart"),
            ],
            metadata={AGENT_METADATA_RESTARTED_KEY: True},
        )
        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                agent_input, turns=20
            )
        # Marker in content of first message from completed run
        all_content = " ".join(m.get("content", "") for m in messages)
        assert PREVIOUS_RUN_MARKER in all_content
        assert "User in completed run" in all_content
        assert "Bot in completed run" in all_content

    def test_build_messages_for_llm_request_restart_partial_run_in_window(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Last N turns only partly in completed run: markers still apply."""
        from rasa.agents.protocol.mcp.mcp_base_agent import (
            END_PREVIOUS_RUN_MARKER,
            PREVIOUS_RUN_MARKER,
        )

        # turns=5 so window is last 5 utterances (tail of run plus new).
        events = [
            UserUttered(text="Old A"),
            BotUttered(text="Old B"),
            UserUttered(text="Old C"),
            AgentStarted(agent_id="my-agent", flow_id="test_flow"),
            UserUttered(text="Run 1"),
            BotUttered(text="Run 2"),
            UserUttered(text="Run 3"),
            BotUttered(text="Run 4"),
            UserUttered(text="Run 5"),
            BotUttered(text="Run 6"),
            AgentCompleted(agent_id="my-agent", flow_id="test_flow"),
            UserUttered(text="After restart"),
        ]
        agent_input = AgentInput(
            id="my-agent",
            user_message="After restart",
            slots=[],
            conversation_history="",
            events=events,
            metadata={AGENT_METADATA_RESTARTED_KEY: True},
        )
        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                agent_input, turns=5
            )
        # Window: Run 4, Run 5, Run 6, After restart
        all_content = " ".join(m.get("content", "") for m in messages)
        assert PREVIOUS_RUN_MARKER in all_content
        assert END_PREVIOUS_RUN_MARKER in all_content
        assert "Run 4" in all_content
        assert "Run 6" in all_content
        assert "After restart" in all_content

    def test_build_messages_for_llm_request_restart_run_outside_window(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Restarted but last N turns are all after the run: no markers in output."""
        from rasa.agents.protocol.mcp.mcp_base_agent import (
            END_PREVIOUS_RUN_MARKER,
            PREVIOUS_RUN_MARKER,
        )

        # Completed run early, then 10+ turns of other interaction. turns=10.
        events = [
            UserUttered(text="Old"),
            BotUttered(text="Old reply"),
            AgentStarted(agent_id="my-agent", flow_id="test_flow"),
            UserUttered(text="In run"),
            BotUttered(text="In run reply"),
            AgentCompleted(agent_id="my-agent", flow_id="test_flow"),
        ]
        for i in range(10):
            events.append(UserUttered(text=f"Later {i}"))
            events.append(BotUttered(text=f"Later reply {i}"))
        events.append(UserUttered(text="Current"))
        agent_input = AgentInput(
            id="my-agent",
            user_message="Current",
            slots=[],
            conversation_history="",
            events=events,
            metadata={AGENT_METADATA_RESTARTED_KEY: True},
        )
        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                agent_input, turns=10
            )
        all_content = " ".join(m.get("content", "") for m in messages)
        assert PREVIOUS_RUN_MARKER not in all_content
        assert END_PREVIOUS_RUN_MARKER not in all_content
        assert "Later 5" in all_content or "Current" in all_content

    def test_build_messages_for_llm_request_restart_no_agent_completed(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """AgentStarted but no AgentCompleted: no markers, no crash."""
        from rasa.agents.protocol.mcp.mcp_base_agent import (
            END_PREVIOUS_RUN_MARKER,
            PREVIOUS_RUN_MARKER,
        )

        events = [
            UserUttered(text="Before"),
            AgentStarted(agent_id="my-agent", flow_id="test_flow"),
            UserUttered(text="In run"),
            BotUttered(text="In run reply"),
            UserUttered(text="Current"),
        ]
        agent_input = AgentInput(
            id="my-agent",
            user_message="Current",
            slots=[],
            conversation_history="",
            events=events,
            metadata={AGENT_METADATA_RESTARTED_KEY: True},
        )
        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                agent_input, turns=20
            )
        all_content = " ".join(m.get("content", "") for m in messages)
        assert PREVIOUS_RUN_MARKER not in all_content
        assert END_PREVIOUS_RUN_MARKER not in all_content
        assert "In run" in all_content
        assert "Current" in all_content

    def test_build_messages_for_llm_request_no_markers_when_not_restarted(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Same events as restart case but restarted=False: no markers in output."""
        from rasa.agents.protocol.mcp.mcp_base_agent import (
            END_PREVIOUS_RUN_MARKER,
            PREVIOUS_RUN_MARKER,
        )

        agent_input = AgentInput(
            id="my-agent",
            user_message="New request after restart",
            slots=[],
            conversation_history="",
            events=[
                UserUttered(text="Prior user"),
                BotUttered(text="Prior bot"),
                AgentStarted(agent_id="my-agent", flow_id="test_flow"),
                UserUttered(text="User in completed run"),
                BotUttered(text="Bot in completed run"),
                AgentCompleted(agent_id="my-agent", flow_id="test_flow"),
                UserUttered(text="New request after restart"),
            ],
            metadata={},  # No RESTARTED_KEY
        )
        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                agent_input, turns=20
            )
        all_content = " ".join(m.get("content", "") for m in messages)
        assert PREVIOUS_RUN_MARKER not in all_content
        assert END_PREVIOUS_RUN_MARKER not in all_content
        assert "User in completed run" in all_content
        assert "New request after restart" in all_content

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
        mock_agent_input.user_message = "How are you?"
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

        # system, user, assistant, user — no extra append since user_message
        # matches the last UserUttered in events
        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "Hello"}
        assert messages[2] == {"role": "assistant", "content": "Hi there"}
        assert messages[3] == {"role": "user", "content": "How are you?"}

    def test_build_messages_for_llm_request_with_buttons(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test that bot messages with buttons include button info in LLM request."""
        mock_agent_input.user_message = "I want to transfer"
        mock_agent_input.events = [
            UserUttered(text="What can I do?"),
            BotUttered(
                text="Please choose an option:",
                data={
                    "buttons": [
                        {"title": "Transfer Money", "payload": "/transfer"},
                        {"title": "Check Balance", "payload": "/balance"},
                    ]
                },
            ),
        ]

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

        assert len(messages) == 4
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "What can I do?"}
        assert "Please choose an option:" in messages[2]["content"]
        assert "Transfer Money" in messages[2]["content"]
        assert "Check Balance" in messages[2]["content"]
        assert messages[3] == {"role": "user", "content": "I want to transfer"}

    def test_build_messages_for_llm_request_handles_empty_events(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test building messages with empty events list."""
        mock_agent_input.user_message = "Hello"
        mock_agent_input.events = []

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System prompt"}
        assert messages[1] == {"role": "user", "content": "Hello"}

    @pytest.mark.parametrize(
        "events,agent_id,expected_start,expected_end",
        [
            ([], "my-agent", None, None),
            (
                [
                    AgentStarted(agent_id="my-agent", flow_id="flow"),
                    UserUttered(text="Hi"),
                ],
                "my-agent",
                None,
                None,
            ),
            (
                [
                    AgentStarted(agent_id="my-agent", flow_id="flow"),
                    UserUttered(text="User"),
                    BotUttered(text="Bot"),
                    AgentCompleted(agent_id="my-agent", flow_id="flow"),
                ],
                "my-agent",
                0,
                3,
            ),
            (
                [
                    AgentStarted(agent_id="other", flow_id="flow"),
                    AgentCompleted(agent_id="other", flow_id="flow"),
                ],
                "my-agent",
                None,
                None,
            ),
            (
                [
                    AgentStarted(agent_id="my-agent", flow_id="flow"),
                    UserUttered(text="First"),
                    AgentCompleted(agent_id="my-agent", flow_id="flow"),
                    AgentStarted(agent_id="my-agent", flow_id="flow"),
                    UserUttered(text="Second"),
                    AgentCompleted(agent_id="my-agent", flow_id="flow"),
                ],
                "my-agent",
                3,
                5,
            ),
        ],
        ids=[
            "empty_events",
            "no_agent_completed",
            "single_run",
            "different_agent_id",
            "two_full_runs",
        ],
    )
    def test_completed_run_region(
        self,
        events: List[Any],
        agent_id: str,
        expected_start: Optional[int],
        expected_end: Optional[int],
    ) -> None:
        """_completed_run_region returns (start_idx, end_idx) for last completed run."""
        start, end = MCPBaseAgent._completed_run_region(events, agent_id)
        assert start == expected_start
        assert end == expected_end
        if expected_start is not None and expected_end is not None:
            assert start is not None and end is not None
            assert events[start].agent_id == agent_id
            assert events[end].agent_id == agent_id

    def test_completed_run_region_two_starts_before_one_completion_uses_earliest_start(
        self,
    ) -> None:
        """Regression: two AgentStarted(agent_id) before one AgentCompleted.

        Region must start at the *first* AgentStarted so the run includes
        the conversation (user/bot utterances). If we used the last
        AgentStarted before completion, the region would be empty of
        utterances and markers would not appear.
        """
        events = [
            AgentStarted(agent_id="my-agent", flow_id="flow"),
            UserUttered(text="User in run"),
            BotUttered(text="Bot in run"),
            AgentStarted(agent_id="my-agent", flow_id="flow"),
            SlotSet(key="x", value=1),
            AgentCompleted(agent_id="my-agent", flow_id="flow"),
        ]
        start, end = MCPBaseAgent._completed_run_region(events, "my-agent")
        assert start == 0
        assert end == 5
        assert events[1].text == "User in run"
        assert events[2].text == "Bot in run"


class TestMCPBaseAgentBuildMessagesContinued(TestMCPBaseAgent):
    """build_messages_for_llm_request tests (inherits TestMCPBaseAgent fixtures)."""

    def test_build_messages_for_llm_request_filters_non_utterance_events(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test that only UserUttered and BotUttered events are used for messages."""
        mock_agent_input.user_message = "Bye"
        mock_agent_input.events = [
            UserUttered(text="Hello"),
            SlotSet("some_slot", "value"),
            BotUttered(text="Hi"),
            UserUttered(text="Bye"),
        ]

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"

            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

        # System + 3 utterance messages (Hello, Hi, Bye) — no extra append
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Bye"

    def test_build_messages_for_llm_request_limits_to_last_n_turns(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Test that only the last N utterance events (turns param) are included."""
        # Build 15 utterance events (more than default turns=10)
        mock_agent_input.events = []
        for i in range(15):
            mock_agent_input.events.append(UserUttered(text=f"User {i}"))
            mock_agent_input.events.append(BotUttered(text=f"Bot {i}"))

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"

            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

        # System + last 10 utterance events + current user_message appended
        # (user_message is not among the windowed events so the guard adds it).
        # Last 10 utterance events are: User 10, Bot 10, ..., User 14, Bot 14
        assert messages[0]["role"] == "system"
        user_and_assistant = [
            m for m in messages[1:] if m["role"] in ("user", "assistant")
        ]
        assert len(user_and_assistant) == 11  # 10 from history + current user_message
        assert user_and_assistant[0]["content"] == "User 10"
        assert user_and_assistant[1]["content"] == "Bot 10"
        assert user_and_assistant[8]["content"] == "User 14"
        assert user_and_assistant[9]["content"] == "Bot 14"
        assert user_and_assistant[10]["content"] == mock_agent_input.user_message

        # Explicit turns=3 keeps only last 3 utterance events + current user_message
        messages_3 = mock_mcp_base_agent.build_messages_for_llm_request(
            mock_agent_input, turns=3
        )
        user_and_assistant_3 = [
            m for m in messages_3[1:] if m["role"] in ("user", "assistant")
        ]
        assert len(user_and_assistant_3) == 4  # 3 from history + current user_message
        assert user_and_assistant_3[0]["content"] == "Bot 13"
        assert user_and_assistant_3[1]["content"] == "User 14"
        assert user_and_assistant_3[2]["content"] == "Bot 14"
        assert user_and_assistant_3[3]["content"] == mock_agent_input.user_message

    def test_build_messages_for_llm_request_no_duplicate_when_filler_message_follows(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Regression: user message must not be duplicated when a filler message.

        BotUttered trails the current UserUttered in events.

        The agent streams an intermediate filler message ("Sure, verifying...")
        and writes it to the tracker before re-invoking.  On re-invocation,
        context.events ends with:

            UserUttered('5272') → BotUttered('Sure, I will now verify...')

        and context.user_message is still '5272'.  The UserUttered is already
        captured by the history loop; the trailing BotUttered must not trick
        the function into appending '5272' a second time.
        """
        mock_agent_input.user_message = "5272"
        mock_agent_input.events = [
            UserUttered(text="Please verify my password"),
            BotUttered(text="Please say your customer password."),
            UserUttered(text="5272"),
            BotUttered(text="Sure, I will now verify your customer password."),
        ]

        with patch.object(mock_mcp_base_agent, "render_prompt_template") as mock_render:
            mock_render.return_value = "System prompt"
            messages = mock_mcp_base_agent.build_messages_for_llm_request(
                mock_agent_input
            )

        user_messages = [m for m in messages if m["role"] == "user"]
        # '5272' must appear exactly once — from the UserUttered event
        assert len(user_messages) == 2
        assert user_messages[0]["content"] == "Please verify my password"
        assert user_messages[1]["content"] == "5272"

        # Full message order: system, user, assistant, user, assistant
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant"]

    def test_build_messages_with_cache_reuses_messages_when_context_unchanged(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """Cache should reuse base messages when context is unchanged."""
        mock_mcp_base_agent._include_date_time = False
        mock_agent_input.events = [
            UserUttered(text="Hi"),
            BotUttered(text="Hello"),
        ]
        cache_state: Dict[str, Any] = {}

        with patch.object(
            mock_mcp_base_agent,
            "build_messages_for_llm_request",
            wraps=mock_mcp_base_agent.build_messages_for_llm_request,
        ) as mock_build_messages:
            first = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )
            second = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )

        assert mock_build_messages.call_count == 1
        assert first == second

    def test_build_messages_with_cache_invalidates_when_slot_changes(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """SlotSet updates from process_tool_output should invalidate cache."""
        mock_mcp_base_agent._include_date_time = False
        mock_mcp_base_agent.prompt_template = "Current user_name: {{ slots.user_name }}"
        mock_agent_input.events = [UserUttered(text="Hi"), BotUttered(text="Hello")]
        cache_state: Dict[str, Any] = {}

        with patch.object(
            mock_mcp_base_agent,
            "build_messages_for_llm_request",
            wraps=mock_mcp_base_agent.build_messages_for_llm_request,
        ) as mock_build_messages:
            _ = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )
            mock_mcp_base_agent._apply_slot_set_events_to_agent_input(
                mock_agent_input, [SlotSet("user_name", "Alice")]
            )
            updated = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )

        assert mock_build_messages.call_count == 2
        assert updated[0]["role"] == "system"
        assert "Alice" in updated[0]["content"]

    def test_build_messages_with_cache_invalidates_when_bot_uttered_event_added(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl, mock_agent_input: AgentInput
    ) -> None:
        """BotUttered from process_tool_output should invalidate conversation cache."""
        mock_mcp_base_agent._include_date_time = False
        mock_agent_input.events = [UserUttered(text="Hi")]
        cache_state: Dict[str, Any] = {}

        with patch.object(
            mock_mcp_base_agent,
            "build_messages_for_llm_request",
            wraps=mock_mcp_base_agent.build_messages_for_llm_request,
        ) as mock_build_messages:
            _ = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )
            mock_agent_input.events.append(BotUttered(text="Processing..."))
            updated = mock_mcp_base_agent._build_messages_for_llm_request_with_cache(
                mock_agent_input,
                cache_state,
            )

        assert mock_build_messages.call_count == 2
        assert any(
            message["role"] == "assistant" and message["content"] == "Processing..."
            for message in updated
        )

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

            mock_execute_mcp.assert_called_once_with(
                "mcp_tool",
                {"arg": "value"},
                None,  # agent_input
            )
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
    async def test_process_tool_output_returns_empty_list(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Test that default process_tool_output returns no events."""
        result = await mock_mcp_base_agent.process_tool_output({}, {})
        assert result == []

    @pytest.mark.asyncio
    async def test_process_tool_output_can_be_overridden(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test custom override of process_tool_output."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        class _CustomMockMCPBaseAgentImpl(MockMCPBaseAgentImpl):
            async def process_tool_output(
                self,
                current_iteration_tool_results: Dict[str, AgentToolResult],
                cumulative_tool_results: Dict[str, AgentToolResult],
                output_channel: Any = None,
            ) -> List:
                return [
                    SlotSet(
                        "tool_result_count",
                        len(current_iteration_tool_results),
                    )
                ]

        agent = _CustomMockMCPBaseAgentImpl(
            name="test_agent",
            description="A test agent",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
        )
        result = await agent.process_tool_output(
            {
                "call_1": AgentToolResult(
                    tool_name="test_tool", result="ok", is_error=False
                )
            },
            {
                "call_1": AgentToolResult(
                    tool_name="test_tool", result="ok", is_error=False
                )
            },
        )
        assert len(result) == 1
        assert isinstance(result[0], SlotSet)
        assert result[0].key == "tool_result_count"
        assert result[0].value == 1

    @pytest.mark.asyncio
    async def test_process_tool_output_or_raise_short_circuits_on_empty_results(
        self, mock_mcp_base_agent: MockMCPBaseAgentImpl
    ) -> None:
        """Short-circuits and returns an empty event list when no results exist."""
        result_events = await mock_mcp_base_agent._process_tool_output_or_raise(
            current_iteration_tool_results={},
            cumulative_tool_results={},
            output_channel=None,
        )

        assert result_events == []

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
        mock_mcp_base_agent._tool_timeout = 0.1

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
            f"{mock_mcp_base_agent._tool_timeout} seconds."
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
        """Test that get_llm_tracing_metadata returns correct metadata.

        Metadata is derived from agent_input.
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

    # ============================================================================
    # Streaming Filler Message Tests
    # ============================================================================

    # ---- _extract_filler_message_from_response --------------------------------

    @pytest.mark.parametrize(
        "choices, tool_calls, expected_text",
        [
            # tool_calls + non-empty content → filler message extracted
            (
                ["I'll look that up for you."],
                [
                    LLMToolCall(
                        id="call_1",
                        type="function",
                        tool_name="search",
                        tool_args={},
                    )
                ],
                "I'll look that up for you.",
            ),
            # tool_calls + whitespace-only content → None
            (
                ["   "],
                [
                    LLMToolCall(
                        id="call_2",
                        type="function",
                        tool_name="search",
                        tool_args={},
                    )
                ],
                None,
            ),
            # tool_calls + empty choices list → None
            (
                [],
                [
                    LLMToolCall(
                        id="call_3",
                        type="function",
                        tool_name="search",
                        tool_args={},
                    )
                ],
                None,
            ),
            # no tool_calls + content → still extracts
            # (method is agnostic to tool_calls)
            (["Just a plain reply."], [], "Just a plain reply."),
        ],
    )
    def test_extract_filler_message_from_response(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        choices: List[str],
        tool_calls: List[LLMToolCall],
        expected_text: Optional[str],
    ) -> None:
        """_extract_filler_message_from_response returns content when present."""
        llm_response = LLMResponse(
            id="resp_1",
            created=1700000000,
            choices=choices,
            tool_calls=tool_calls,
        )

        result = mock_mcp_base_agent._extract_filler_message_from_response(llm_response)

        assert result == expected_text

    # ---- _send_filler_message: guard conditions ----------------------

    @pytest.mark.asyncio
    async def test_send_filler_message_skips_when_no_output_channel(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """No output channel → method returns early, no event appended."""
        mock_agent_input.recipient_id = "user_123"
        generated_events: List = []

        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="On it!",
            output_channel=None,
            generated_events=generated_events,
        )

        assert generated_events == []

    @pytest.mark.asyncio
    async def test_send_filler_message_skips_when_empty_text(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """Empty / whitespace filler message text → method returns early."""
        mock_agent_input.recipient_id = "user_123"
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()
        generated_events: List = []

        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="   ",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        mock_channel.send_text_message.assert_not_called()
        assert generated_events == []

    @pytest.mark.asyncio
    async def test_send_filler_message_skips_when_no_recipient_id(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """No recipient_id and no sender_id in metadata → method returns early."""
        mock_agent_input.recipient_id = None
        mock_agent_input.metadata = {}  # no AGENT_METADATA_SENDER_ID_KEY
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()
        generated_events: List = []

        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="On it!",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        mock_channel.send_text_message.assert_not_called()
        assert generated_events == []

    # ---- _send_filler_message: fallback (non-streaming) path ---------

    @pytest.mark.asyncio
    async def test_send_filler_message_uses_send_text_message_when_no_streaming(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """When supports_streaming is False, send_text_message is called."""
        mock_agent_input.recipient_id = "user_abc"
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()
        generated_events: List = []

        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="Sure, let me check!",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        mock_channel.send_text_message.assert_awaited_once_with(
            recipient_id="user_abc",
            text="Sure, let me check!",
        )
        # Streaming methods must NOT be called
        mock_channel.send_response_chunk_start.assert_not_called()
        mock_channel.send_response_chunk.assert_not_called()
        mock_channel.send_response_chunk_end.assert_not_called()

    # ---- _send_filler_message: streaming path ------------------------

    @pytest.mark.asyncio
    async def test_send_filler_message_uses_streaming_when_supported(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """When supports_streaming is True, chunk streaming methods are called."""
        mock_agent_input.recipient_id = "user_xyz"
        mock_channel = MagicMock()
        mock_channel.supports_streaming = True
        mock_channel.send_response_chunk_start = AsyncMock()
        mock_channel.send_response_chunk = AsyncMock()
        mock_channel.send_response_chunk_end = AsyncMock()
        generated_events: List = []

        filler_text = "On it!"
        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text=filler_text,
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        mock_channel.send_response_chunk_start.assert_awaited_once_with("user_xyz")
        mock_channel.send_response_chunk_end.assert_awaited_once_with(
            "user_xyz", is_intermediate=True
        )
        # send_text_message must NOT be called
        mock_channel.send_text_message.assert_not_called()

        # All chunks concatenated must equal the original text
        sent_chunks = [
            c.kwargs["chunk"] for c in mock_channel.send_response_chunk.await_args_list
        ]
        assert "".join(sent_chunks) == filler_text

    @pytest.mark.asyncio
    async def test_stream_filler_message_chunks_respects_chunk_size(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
    ) -> None:
        """_stream_filler_message_chunks splits text into CHUNK_SIZE-sized pieces."""
        mock_channel = MagicMock()
        mock_channel.send_response_chunk_start = AsyncMock()
        mock_channel.send_response_chunk = AsyncMock()
        mock_channel.send_response_chunk_end = AsyncMock()

        # Text longer than one chunk
        text = "A" * (AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE * 3 + 2)
        await mock_mcp_base_agent._stream_filler_message_chunks(
            mock_channel, "recipient_1", text
        )

        chunks = [
            c.kwargs["chunk"] for c in mock_channel.send_response_chunk.await_args_list
        ]
        assert "".join(chunks) == text
        # Every chunk except possibly the last must be exactly CHUNK_SIZE characters
        for chunk in chunks[:-1]:
            assert len(chunk) == AGENT_FILLER_MESSAGE_STREAM_DEFAULT_CHUNK_SIZE

    # ---- BotUttered event appended to generated_events ----------------------

    @pytest.mark.asyncio
    async def test_send_filler_message_appends_bot_uttered_event(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """A BotUttered event is appended to generated_events after sending."""
        mock_agent_input.recipient_id = "user_evt"
        mock_agent_input.metadata = {
            AGENT_METADATA_AGENT_ID_KEY: "agent_42",
            AGENT_METADATA_MODEL_ID_KEY: "gpt-4o",
        }
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()
        generated_events: List = []

        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="Working on it…",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        assert len(generated_events) == 1
        event = generated_events[0]
        assert isinstance(event, BotUttered)
        assert event.text == "Working on it…"

        meta = event.metadata
        assert meta["message_type"] == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE
        assert meta[AGENT_METADATA_AGENT_ID_KEY] == "agent_42"
        assert meta[AGENT_METADATA_MODEL_ID_KEY] == "gpt-4o"
        assert meta["agent_name"] == mock_mcp_base_agent._name

    @pytest.mark.asyncio
    async def test_send_filler_message_no_event_on_channel_error(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """If the channel raises an exception, no BotUttered event is appended."""
        mock_agent_input.recipient_id = "user_err"
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock(
            side_effect=RuntimeError("channel down")
        )
        generated_events: List = []

        # Should not propagate the exception
        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="Hang on!",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        assert generated_events == []

    # ---- recipient_id resolution ---------------------------------------------

    @pytest.mark.parametrize(
        "recipient_id, metadata, expected",
        [
            ("direct_id", {}, "direct_id"),
            (None, {AGENT_METADATA_SENDER_ID_KEY: "meta_id"}, "meta_id"),
            (None, {}, None),
        ],
    )
    def test_get_recipient_id(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
        recipient_id: Optional[str],
        metadata: Dict[str, Any],
        expected: Optional[str],
    ) -> None:
        """_get_recipient_id prefers recipient_id, falls back to metadata sender_id."""
        mock_agent_input.recipient_id = recipient_id
        mock_agent_input.metadata = metadata

        result = mock_mcp_base_agent._get_recipient_id(mock_agent_input)

        assert result == expected

    # ---- _create_filler_message_metadata -------------------------------------

    def test_create_filler_message_metadata_structure(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """_create_filler_message_metadata returns expected keys and values."""
        mock_agent_input.metadata = {
            AGENT_METADATA_AGENT_ID_KEY: "agent_99",
            AGENT_METADATA_MODEL_ID_KEY: "gpt-4o-mini",
        }

        meta = mock_mcp_base_agent._create_filler_message_metadata(mock_agent_input)

        assert meta["message_type"] == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE
        assert meta["agent_name"] == mock_mcp_base_agent._name
        assert meta["utter_source"] == mock_mcp_base_agent.__class__.__name__
        assert meta[AGENT_METADATA_AGENT_ID_KEY] == "agent_99"
        assert meta[AGENT_METADATA_MODEL_ID_KEY] == "gpt-4o-mini"

    # ---- Integration: filler message events surface in AgentOutput -----------

    @pytest.mark.asyncio
    async def test_filler_message_events_included_in_agent_output_events(
        self,
        mock_mcp_base_agent: MockMCPBaseAgentImpl,
        mock_agent_input: AgentInput,
    ) -> None:
        """BotUttered filler message events are included in AgentOutput.events.

        This test simulates the pattern used by send_message implementations:
        a generated_events list is populated by _send_filler_message and
        then passed through to AgentOutput.events.
        """
        from rasa.agents.core.types import AgentStatus
        from rasa.agents.schemas import AgentOutput

        mock_agent_input.recipient_id = "user_out"
        mock_agent_input.metadata = {
            AGENT_METADATA_AGENT_ID_KEY: "agent_out",
            AGENT_METADATA_MODEL_ID_KEY: "gpt-4o",
        }
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()

        generated_events: List = []

        # Simulate what send_message does: call _send_filler_message,
        # then build AgentOutput with the accumulated events.
        await mock_mcp_base_agent._send_filler_message(
            agent_input=mock_agent_input,
            filler_message_text="Let me check that for you.",
            output_channel=mock_channel,
            generated_events=generated_events,
        )

        agent_output = AgentOutput(
            id=mock_agent_input.id,
            status=AgentStatus.INPUT_REQUIRED,
            response_message="Here is the answer.",
            events=generated_events if generated_events else None,
        )

        assert agent_output.events is not None
        assert len(agent_output.events) == 1
        filler_event = agent_output.events[0]
        assert isinstance(filler_event, BotUttered)
        assert filler_event.text == "Let me check that for you."
        assert (
            filler_event.metadata["message_type"]
            == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FILLER_MESSAGE
        )

    # ============================================================================
    # Tool Timeout Configuration Tests
    # ============================================================================

    def test_init_with_default_tool_timeout(self, monkeypatch: MonkeyPatch) -> None:
        """Test that default tool timeout is used when not specified."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl(
            name="test_agent",
            description="Test description",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
        )

        assert agent._tool_timeout == MCPBaseAgent.TOOL_CALL_DEFAULT_TIMEOUT
        assert agent._tool_timeout == 10

    def test_init_with_custom_tool_timeout(self, monkeypatch: MonkeyPatch) -> None:
        """Test that custom tool timeout is used when specified."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl(
            name="test_agent",
            description="Test description",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
            tool_timeout=30,
        )

        assert agent._tool_timeout == 30

    def test_init_with_zero_tool_timeout_raises(self, monkeypatch: MonkeyPatch) -> None:
        """Test that zero tool timeout is rejected."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        with pytest.raises(ValueError, match="tool_timeout"):
            MockMCPBaseAgentImpl(
                name="test_agent",
                description="Test description",
                protocol_type=ProtocolConfig.RASA,
                server_configs=[],
                tool_timeout=0,
            )

    def test_init_with_negative_tool_timeout_raises(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that negative tool timeout is rejected."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        with pytest.raises(ValueError, match="tool_timeout"):
            MockMCPBaseAgentImpl(
                name="test_agent",
                description="Test description",
                protocol_type=ProtocolConfig.RASA,
                server_configs=[],
                tool_timeout=-1,
            )

    @pytest.mark.parametrize(
        "tool_timeout", [float("nan"), float("inf"), float("-inf")]
    )
    def test_init_with_non_finite_tool_timeout_raises(
        self, monkeypatch: MonkeyPatch, tool_timeout: float
    ) -> None:
        """Test that NaN and infinities are rejected for tool timeout."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        with pytest.raises(ValueError, match="finite number"):
            MockMCPBaseAgentImpl(
                name="test_agent",
                description="Test description",
                protocol_type=ProtocolConfig.RASA,
                server_configs=[],
                tool_timeout=tool_timeout,
            )

    def test_from_config_with_tool_timeout(self, monkeypatch: MonkeyPatch) -> None:
        """Test from_config uses tool_timeout from configuration."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="Test agent",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                llm={"provider": "openai", "model": "gpt-4"},
                tool_timeout=45,
            ),
            connections=AgentConnections(),
        )

        agent = MockMCPBaseAgentImpl.from_config(agent_config)

        assert agent._tool_timeout == 45

    def test_from_config_without_tool_timeout(self, monkeypatch: MonkeyPatch) -> None:
        """Test from_config uses default tool timeout when not in configuration."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_agent",
                description="Test agent",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                llm={"provider": "openai", "model": "gpt-4"},
            ),
            connections=AgentConnections(),
        )

        agent = MockMCPBaseAgentImpl.from_config(agent_config)

        assert agent._tool_timeout == MCPBaseAgent.TOOL_CALL_DEFAULT_TIMEOUT
        assert agent._tool_timeout == 10

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_uses_configured_timeout(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that _execute_mcp_tool uses the configured tool_timeout."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl(
            name="test_agent",
            description="Test description",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
            tool_timeout=25,
        )

        mock_connection = MagicMock()
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value={"result": "success"})
        mock_connection.ensure_active_session = AsyncMock(return_value=mock_session)
        mock_connection.server_url = "http://localhost:8000"

        agent._tool_to_server_mapper["test_tool"] = "test_server"
        agent._server_connections["test_server"] = mock_connection

        with patch(
            "rasa.agents.schemas.AgentToolResult.from_mcp_tool_result"
        ) as mock_from_mcp:
            mock_from_mcp.return_value = AgentToolResult(
                tool_name="test_tool",
                result='{"result": "success"}',
                is_error=False,
            )

            await agent._execute_mcp_tool("test_tool", {"arg": "value"})

            mock_session.call_tool.assert_called_once_with(
                "test_tool",
                {"arg": "value"},
                read_timeout_seconds=timedelta(seconds=25),
            )

    @pytest.mark.asyncio
    async def test_custom_tool_uses_configured_timeout(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that custom tools use the configured tool_timeout."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key")

        agent = MockMCPBaseAgentImpl(
            name="test_agent",
            description="Test description",
            protocol_type=ProtocolConfig.RASA,
            server_configs=[],
            tool_timeout=0.2,
        )

        mock_custom_tool = MagicMock()
        mock_custom_tool.tool_name = "slow_tool"

        async def slow_tool_executor(args):
            await anyio.sleep(0.3)
            return AgentToolResult(
                tool_name="slow_tool",
                result="result",
                is_error=False,
            )

        mock_custom_tool.tool_executor = slow_tool_executor
        agent._custom_tools = [mock_custom_tool]

        result = await agent._execute_tool_call("slow_tool", {"arg": "value"})

        assert result.tool_name == "slow_tool"
        assert result.is_error is True
        assert "timed out after" in result.error_message
        assert "0.2 seconds" in result.error_message
