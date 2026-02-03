"""Tests for AgentCopilot class."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
from rasa.builder.copilot.exceptions import CopilotNextStreamEventTimeoutException
from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    TextContent,
    UserChatMessage,
)
from rasa.builder.shared.tracker_context import CurrentState, TrackerContext


class TestAgentCopilot:
    """Test class for AgentCopilot."""

    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock config values for testing."""
        monkeypatch.setattr("rasa.builder.config.USE_AGENT_SDK_COPILOT", True)
        monkeypatch.setattr("rasa.builder.config.OPENAI_MODEL", "gpt-4")
        monkeypatch.setattr("rasa.builder.config.OPENAI_TEMPERATURE", 0.7)
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_PORT", 5051)
        monkeypatch.setattr("rasa.builder.config.MCP_TOOL_CALL_TIMEOUT", 30)
        monkeypatch.setattr("rasa.builder.config.MCP_MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr("rasa.builder.config.COPILOT_INPUT_TOKEN_PRICE", 0.001)
        monkeypatch.setattr("rasa.builder.config.COPILOT_OUTPUT_TOKEN_PRICE", 0.002)
        monkeypatch.setattr("rasa.builder.config.COPILOT_CACHED_TOKEN_PRICE", 0.0005)

    @pytest.fixture
    def sample_context(self) -> CopilotContext:
        """Create a sample copilot context."""
        return CopilotContext(
            tracker_context=TrackerContext(
                conversation_turns=[],
                current_state=CurrentState(),
            ),
            assistant_logs="Some assistant logs",
            assistant_files={"domain.yml": "version: '3.1'"},
            copilot_chat_history=[
                UserChatMessage(
                    role="user",
                    content=[TextContent(type="text", text="How do I create a flow?")],
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_init(self, mock_config):
        """Test AgentCopilot initialization."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()

        assert copilot._system_message_prompt_template is not None
        assert copilot.usage_statistics is not None
        # Token counts are None until a generation is performed
        assert copilot.usage_statistics.prompt_tokens is None
        assert copilot.usage_statistics.completion_tokens is None

    @pytest.mark.asyncio
    async def test_usage_statistics_property(self, mock_config):
        """Test usage_statistics property returns usage statistics."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        stats = copilot.usage_statistics

        assert stats is not None
        assert hasattr(stats, "prompt_tokens")
        assert hasattr(stats, "completion_tokens")
        assert hasattr(stats, "total_tokens")

    @pytest.mark.asyncio
    async def test_llm_config_property(self, mock_config):
        """Test llm_config property returns correct configuration."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        config = copilot.llm_config

        assert config["model"] == "gpt-4"
        assert config["temperature"] == 0.7
        assert config["stream"] is True
        assert config["stream_options"]["include_usage"] is True

    @pytest.mark.asyncio
    async def test_build_messages_empty_history(self, mock_config):
        """Test _build_messages with empty chat history."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        context = CopilotContext(
            tracker_context=None,
            assistant_logs="",
            assistant_files={},
            copilot_chat_history=[],
        )

        messages = await copilot._build_messages(context)

        assert messages == []

    @pytest.mark.asyncio
    async def test_build_messages_with_history(self, mock_config, sample_context):
        """Test _build_messages with chat history."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        messages = await copilot._build_messages(sample_context)

        assert len(messages) >= 1
        # Last message should be the user's message
        assert messages[-1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_convert_to_responses_api_format_simple_string(self, mock_config):
        """Test _convert_to_responses_api_format with simple string content."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        converted = copilot._convert_to_responses_api_format(messages)

        assert len(converted) == 2
        assert converted[0] == {"role": "user", "content": "Hello"}
        assert converted[1] == {"role": "assistant", "content": "Hi there!"}

    @pytest.mark.asyncio
    async def test_convert_to_responses_api_format_list_content(self, mock_config):
        """Test _convert_to_responses_api_format with list content."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
        ]

        converted = copilot._convert_to_responses_api_format(messages)

        assert len(converted) == 2
        # User messages should have input_text type
        assert converted[0]["content"][0]["type"] == "input_text"
        assert converted[0]["content"][0]["text"] == "Hello"
        # Assistant messages should have output_text type
        assert converted[1]["content"][0]["type"] == "output_text"
        assert converted[1]["content"][0]["text"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_generate_response_returns_handler_and_context(
        self, mock_config, sample_context
    ):
        """Test generate_response returns handler and generation context."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
        from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (  # noqa: E501
            AgentCopilotResponseHandler,
        )

        # Mock the _stream_response method to avoid MCP server connection
        async def mock_stream():
            yield MagicMock()

        with patch.object(AgentCopilot, "_stream_response", return_value=mock_stream()):
            copilot = AgentCopilot()
            handler, gen_context = await copilot.generate_response(sample_context)

            assert isinstance(handler, AgentCopilotResponseHandler)
            assert isinstance(gen_context, CopilotGenerationContext)
            assert gen_context.system_message is not None
            assert gen_context.last_user_message is not None

    @pytest.mark.asyncio
    async def test_generate_response_resets_usage_statistics(
        self, mock_config, sample_context
    ):
        """Test that generate_response resets usage statistics."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        async def mock_stream():
            yield MagicMock()

        with patch.object(AgentCopilot, "_stream_response", return_value=mock_stream()):
            copilot = AgentCopilot()
            # Set some values first
            copilot._usage_statistics.prompt_tokens = 100
            copilot._usage_statistics.completion_tokens = 50

            await copilot.generate_response(sample_context)

            # Usage should be reset to None (default state)
            assert copilot.usage_statistics.prompt_tokens is None
            assert copilot.usage_statistics.completion_tokens is None

    @pytest.mark.asyncio
    @patch("rasa.builder.copilot.agent_sdk.agent_copilot.Runner.run_streamed")
    @patch.object(AgentCopilot, "_create_agent")
    async def test_stream_response_times_out_when_agent_stream_stalls(
        self, mock_create_agent, mock_run_streamed
    ):
        """Test that AgentCopilot fails fast when no new stream events arrive."""

        # Given
        @asynccontextmanager
        async def fake_create_agent(system_instructions):
            yield MagicMock()

        async def stalled_events():
            yield MagicMock()
            await asyncio.Event().wait()

        fake_result = MagicMock()
        fake_result.stream_events.return_value = stalled_events()

        mock_run_streamed.return_value = fake_result
        mock_create_agent.side_effect = fake_create_agent

        copilot = AgentCopilot()
        copilot._timeout_for_next_stream_event = 0.1
        stream = copilot._stream_response(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hi"}],
        )

        # When
        # First event is yielded
        await anext(stream)

        # Then
        # Second event stalls and should time out
        with pytest.raises(CopilotNextStreamEventTimeoutException):
            await anext(stream)


class TestAgentCopilotMCPConnection:
    """Test MCP server connection handling."""

    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock config values for testing."""
        monkeypatch.setattr("rasa.builder.config.USE_AGENT_SDK_COPILOT", True)
        monkeypatch.setattr("rasa.builder.config.OPENAI_MODEL", "gpt-4")
        monkeypatch.setattr("rasa.builder.config.OPENAI_TEMPERATURE", 0.7)
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_HOST", "127.0.0.1")
        monkeypatch.setattr("rasa.builder.config.MCP_SERVER_PORT", 5051)
        monkeypatch.setattr("rasa.builder.config.MCP_TOOL_CALL_TIMEOUT", 30)
        monkeypatch.setattr("rasa.builder.config.MCP_MAX_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr("rasa.builder.config.COPILOT_INPUT_TOKEN_PRICE", 0.001)
        monkeypatch.setattr("rasa.builder.config.COPILOT_OUTPUT_TOKEN_PRICE", 0.002)
        monkeypatch.setattr("rasa.builder.config.COPILOT_CACHED_TOKEN_PRICE", 0.0005)

    @pytest.mark.asyncio
    async def test_create_mcp_server_context_manager(self, mock_config):
        """Test _create_mcp_server creates proper context manager."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()

        # Mock TracedMCPServerWrapper
        mock_server = MagicMock()
        mock_server.__aenter__ = AsyncMock(return_value=mock_server)
        mock_server.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "rasa.builder.copilot.agent_sdk.agent_copilot.TracedMCPServerWrapper",
            return_value=mock_server,
        ):
            async with copilot._create_mcp_server() as server:
                assert server is mock_server

    @pytest.mark.asyncio
    async def test_create_mcp_server_connection_error(self, mock_config):
        """Test _create_mcp_server handles connection errors."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()

        # Mock TracedMCPServerWrapper to raise exception
        mock_server = MagicMock()
        mock_server.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "rasa.builder.copilot.agent_sdk.agent_copilot.TracedMCPServerWrapper",
            return_value=mock_server,
        ):
            with pytest.raises(Exception, match="Connection failed"):
                async with copilot._create_mcp_server():
                    pass

    @pytest.mark.asyncio
    async def test_create_agent_context_manager(self, mock_config):
        """Test _create_agent creates agent with MCP server."""
        from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot

        copilot = AgentCopilot()

        # Mock the MCP server context manager
        mock_server = MagicMock()

        # Mock Agent class
        mock_agent = MagicMock()

        with (
            patch.object(
                copilot,
                "_create_mcp_server",
                return_value=self._async_context_manager(mock_server),
            ),
            patch(
                "rasa.builder.copilot.agent_sdk.agent_copilot.Agent",
                return_value=mock_agent,
            ) as mock_agent_class,
        ):
            async with copilot._create_agent("Test instructions") as agent:
                assert agent is mock_agent

            # Verify Agent was created with correct params
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs["name"] == "Rasa Copilot"
            assert call_kwargs["instructions"] == "Test instructions"
            assert call_kwargs["model"] == "gpt-4"
            assert mock_server in call_kwargs["mcp_servers"]

    @staticmethod
    def _async_context_manager(value):
        """Helper to create async context manager."""
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def cm():
            yield value

        return cm()
