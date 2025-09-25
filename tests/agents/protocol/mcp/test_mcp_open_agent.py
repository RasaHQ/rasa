"""Unit tests for MCPOpenAgent."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rasa.agents.protocol.mcp.mcp_open_agent import MCPOpenAgent
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentToolResult
from rasa.core.available_agents import (
    AgentConfig,
    AgentInfo,
    ProtocolConfig,
)
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall


class TestMCPOpenAgent:
    """Test cases for MCPOpenAgent."""

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
    def mcp_open_agent(self, monkeypatch: pytest.MonkeyPatch) -> MCPOpenAgent:
        """Fixture for creating an MCPOpenAgent instance."""
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_mcp_open_agent")
        return MCPOpenAgent.from_config(
            AgentConfig(
                agent=AgentInfo(
                    name="test_open_agent",
                    description="A test open agent for unit testing",
                    protocol=ProtocolConfig.RASA,
                )
            )
        )

    def test_render_prompt_template_basic_rendering(
        self, mcp_open_agent: MCPOpenAgent, mock_agent_input: AgentInput
    ):
        """Test basic prompt template rendering with all context variables."""
        with patch("rasa.agents.protocol.mcp.mcp_base_agent.datetime") as mock_datetime:
            # Mock the current datetime
            mock_now = datetime(2024, 1, 15, 14, 30, 45)  # Monday, 2:30:45 PM
            mock_datetime.now.return_value = mock_now

            result = mcp_open_agent.render_prompt_template(mock_agent_input)

            # Verify the template was rendered with correct date/time values
            assert "2024-01-15" in result  # current_date
            assert "14:30:45" in result  # current_time
            assert "Monday" in result  # current_day

            # Verify other context variables are included
            assert "A test open agent for unit testing" in result  # description
            assert "Previous conversation..." in result  # conversation_history

            # Verify template structure is maintained (MCP Open Agent template)
            assert "### Context" in result
            assert "### Primary Task" in result
            assert "### Instructions" in result
            assert "### Conversation history" in result

    def test_render_prompt_template_excludes_specified_fields(
        self, mcp_open_agent: MCPOpenAgent, mock_agent_input: AgentInput
    ):
        """Test that render_prompt_template excludes some fields from context."""
        with patch("rasa.agents.protocol.mcp.mcp_base_agent.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 15, 14, 30, 45)
            mock_datetime.now.return_value = mock_now

            result = mcp_open_agent.render_prompt_template(mock_agent_input)

            # Verify excluded fields are not in the rendered template
            assert "test_id" not in result  # id should be excluded
            assert "2024-01-15T10:30:00Z" not in result  # timestamp should be excluded

    def test_get_task_completed_tool(self):
        """Test getting the task completed tool."""
        tool = MCPOpenAgent.get_task_completed_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "task_completed"
        assert "FULLY completed" in tool["function"]["description"]
        assert "message" in tool["function"]["parameters"]["properties"]
        assert tool["function"]["parameters"]["required"] == ["message"]

    def test_get_agent_specific_built_in_tools(self, mock_agent_input):
        """Test getting agent-specific built-in tools."""
        tools = MCPOpenAgent.get_agent_specific_built_in_tools(mock_agent_input)

        assert len(tools) == 1
        assert tools[0].name == "task_completed"

    @pytest.mark.parametrize(
        "tool_args, expected_message",
        [
            ({"message": "Task completed successfully"}, "Task completed successfully"),
            ({}, "Task completed"),  # Default message
        ],
    )
    def test_run_task_completed_tool(
        self, mcp_open_agent, mock_agent_input, tool_args, expected_message
    ):
        """Test running the task completed tool with various message configurations."""

        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="task_completed",
            tool_args=tool_args,
        )

        tool_results = {}

        result = mcp_open_agent._run_task_completed_tool(
            tool_call, mock_agent_input, tool_results
        )

        assert result.id == mock_agent_input.id
        assert result.status.name == "COMPLETED"
        assert result.response_message == expected_message
        assert "call_123" in tool_results
        assert tool_results["call_123"].tool_name == "task_completed"
        assert tool_results["call_123"].result == expected_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_response, expected_status, expected_message_keyword",
        [
            # No LLM response
            (
                LLMResponse(
                    id="test_id", created=1642248600, choices=[], tool_calls=None
                ),
                "RECOVERABLE_ERROR",
                "No response from LLM",
            ),
            # No tool calls
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["Test response"],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                "Test response",
            ),
        ],
    )
    async def test_send_message_llm_response_scenarios(
        self,
        mcp_open_agent,
        mock_agent_input,
        llm_response,
        expected_status,
        expected_message_keyword,
    ):
        """Test send_message with various LLM response scenarios."""
        with patch.object(mcp_open_agent, "llm_client") as mock_llm_client:
            mock_llm_client.acompletion = AsyncMock(return_value=llm_response)

            result = await mcp_open_agent.send_message(mock_agent_input)

            assert result.id == mock_agent_input.id
            assert result.status.name == expected_status
            if expected_status == "RECOVERABLE_ERROR":
                assert expected_message_keyword in result.error_message
            else:
                assert result.response_message == expected_message_keyword

    @pytest.mark.asyncio
    async def test_send_message_task_completed_tool(
        self, mcp_open_agent, mock_agent_input
    ):
        """Test send_message with task completed tool call."""

        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="task_completed",
            tool_args={"message": "Task completed successfully"},
        )

        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Task completed successfully"],
            tool_calls=[mock_tool_call],
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = [MagicMock(name="task_completed")]

            result = await mcp_open_agent.send_message(mock_agent_input)

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert result.response_message == "Task completed successfully"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name, tool_args, is_error, expected_behavior",
        [
            # Other tool call (not task_completed)
            ("other_tool", {"arg": "value"}, False, "continue_processing"),
            # Tool call failure
            ("other_tool", {"arg": "value"}, True, "error_output"),
        ],
    )
    async def test_send_message_tool_call_scenarios(
        self,
        mcp_open_agent,
        mock_agent_input,
        tool_name,
        tool_args,
        is_error,
        expected_behavior,
    ):
        """Test send_message with various tool call scenarios."""
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name=tool_name,
            tool_args=tool_args,
        )

        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed successfully"],
            tool_calls=[mock_tool_call],
        )

        mock_tool_output = AgentToolResult(
            tool_name=tool_name,
            result="Tool result",
            is_error=is_error,
        )

        # Set max iterations to 1 to force completion
        mcp_open_agent.MAX_ITERATIONS = 1

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent, "_generate_agent_error_output"
            ) as mock_error_output,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = [MagicMock(name=tool_name)]
            mock_execute_tool.return_value = mock_tool_output
            mock_error_output.return_value = MagicMock()

            await mcp_open_agent.send_message(mock_agent_input)

            if expected_behavior == "continue_processing":
                mock_execute_tool.assert_called_once_with(tool_name, tool_args)
            elif expected_behavior == "error_output":
                mock_error_output.assert_called_once_with(
                    mock_tool_output, mock_agent_input, mock_tool_call
                )

    @pytest.mark.asyncio
    async def test_send_message_malformed_tool_response_retry(
        self, mcp_open_agent, mock_agent_input
    ):
        """Test send_message with malformed tool response that triggers retry."""

        # Create a proper exception with original_exception attribute
        decode_error = LLMToolResponseDecodeError("Invalid JSON")
        provider_exception = ProviderClientAPIException("Decode error")
        provider_exception.original_exception = decode_error

        mock_llm_client = MagicMock()
        mock_llm_client.acompletion.side_effect = [
            provider_exception,
            LLMResponse(
                id="test_id",
                created=1642248600,
                choices=["Success response"],
                tool_calls=[],
            ),
        ]
        mcp_open_agent.llm_client = mock_llm_client

        with patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools:
            mock_get_tools.return_value = []

            result = await mcp_open_agent.send_message(mock_agent_input)

            assert result.id == mock_agent_input.id
            # We want to see if the second LLM call is made.
            assert mock_llm_client.acompletion.call_count == 2

            # Check that the 2nd call includes the system message
            # for malformed tool response
            second_call_args = mock_llm_client.acompletion.call_args_list[1]
            messages = second_call_args[0][0]  # First positional argument

            # Should have 3 messages
            # system, user, and malformed tool response system message
            assert len(messages) == 3
            assert messages[0]["role"] == "system"  # Original system message
            assert messages[1]["role"] == "user"  # User message
            assert messages[2]["role"] == "system"  # Malformed tool response message

            # System message for malformed tool response
            system_message = (
                "The previous tool response contained invalid or incomplete JSON and "
                "could not be parsed. Retry by generating a tool response in STRICT "
                "JSON string format only. Ensure the JSON is fully well-formed and "
                "corresponds exactly to the user's last request."
            )
            assert messages[2]["content"] == system_message

    @pytest.mark.asyncio
    async def test_send_message_general_exception(
        self, mcp_open_agent, mock_agent_input
    ):
        """Test send_message with general exception."""
        with patch.object(mcp_open_agent, "llm_client") as mock_llm_client:
            mock_llm_client.acompletion.side_effect = Exception("General error")

            result = await mcp_open_agent.send_message(mock_agent_input)

            assert result.id == mock_agent_input.id
            assert result.status.name == "FATAL_ERROR"
            assert "I encountered an error" in result.response_message
            assert "General error" in result.error_message

    @pytest.mark.asyncio
    async def test_send_message_max_iterations_reached(
        self, mcp_open_agent, mock_agent_input
    ):
        """Test send_message when max iterations are reached."""
        # Create a tool call that will keep the agent in a loop
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )

        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[mock_tool_call],
        )

        mock_tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
        ):
            # Always return the same response to create an infinite loop
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = [MagicMock(name="other_tool")]
            mock_execute_tool.return_value = mock_tool_output

            # Set max iterations to 1 to force completion
            mcp_open_agent.MAX_ITERATIONS = 1

            result = await mcp_open_agent.send_message(mock_agent_input)

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert "couldn't provide a final answer" in result.response_message
