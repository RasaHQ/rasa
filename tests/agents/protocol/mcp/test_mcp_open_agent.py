"""Unit tests for MCPOpenAgent."""

from datetime import datetime
from typing import Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from rasa.agents.constants import (
    AGENT_METADATA_AGENT_ID_KEY,
    AGENT_METADATA_MODEL_ID_KEY,
    AGENT_METADATA_SENDER_ID_KEY,
)
from rasa.agents.protocol.mcp.mcp_open_agent import KEY_TASK_COMPLETED, MCPOpenAgent
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentToolResult
from rasa.core.available_agents import (
    AgentConfig,
    AgentConfiguration,
    AgentInfo,
    ProtocolConfig,
)
from rasa.core.constants import (
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE,
)
from rasa.shared.constants import (
    DEFAULT_INCLUDE_DATE_TIME,
    DEFAULT_TIMEZONE,
    OPENAI_API_KEY_ENV_VAR,
)
from rasa.shared.core.events import BotUttered, SlotSet
from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall


@pytest.fixture
def mcp_open_agent(monkeypatch: pytest.MonkeyPatch) -> MCPOpenAgent:
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
    def mock_output_channel(self):
        """Non-streaming output channel fixture (LLM called via acompletion)."""
        channel = MagicMock()
        channel.supports_streaming = False
        channel.send_text_message = AsyncMock()
        return channel

    def test_render_prompt_template_basic_rendering(
        self, mcp_open_agent: MCPOpenAgent, mock_agent_input: AgentInput
    ):
        """Test basic prompt template rendering with all context variables."""
        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC"))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            result = mcp_open_agent.render_prompt_template(mock_agent_input)

            # Verify the template was rendered with correct date/time values
            assert "- Current date: 15 January, 2024" in result  # current_date
            assert "- Current time: 14:30:45 (UTC)" in result  # current_time
            assert "- Current day: Monday" in result  # current_day

            # Verify other context variables are included
            assert "A test open agent for unit testing" in result  # description

            # Verify template structure is maintained (MCP Open Agent template)
            assert "### Date & Time Context" in result
            assert "### Primary Task" in result
            assert "### Instructions" in result

    def test_render_prompt_template_excludes_specified_fields(
        self, mcp_open_agent: MCPOpenAgent, mock_agent_input: AgentInput
    ):
        """Test that render_prompt_template excludes some fields from context."""
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = datetime(
                2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo("UTC")
            )

            result = mcp_open_agent.render_prompt_template(mock_agent_input)

            # Verify excluded fields are not in the rendered template
            assert "test_id" not in result  # id should be excluded
            assert "2024-01-15T10:30:00Z" not in result  # timestamp should be excluded

    @pytest.mark.parametrize(
        "include_date_time, timezone, expected_datetime_present, expected_date_format,"
        "expected_time_format, expected_day",
        [
            # include_date_time is True (default), should include datetime
            (
                DEFAULT_INCLUDE_DATE_TIME,
                DEFAULT_TIMEZONE,
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is True with custom timezone
            (
                True,
                "America/New_York",
                True,
                "15 January, 2024",
                "14:30:45",
                "Monday",
            ),
            # include_date_time is False
            # should NOT include datetime
            (False, DEFAULT_TIMEZONE, False, None, None, None),
            # include_date_time is False with custom timezone
            # should NOT include datetime
            (False, "America/New_York", False, None, None, None),
        ],
    )
    def test_render_prompt_template_includes_current_datetime_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_agent_input: AgentInput,
        include_date_time: bool,
        timezone: str,
        expected_datetime_present: bool,
        expected_date_format: Optional[str],
        expected_time_format: Optional[str],
        expected_day: Optional[str],
    ):
        monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_mcp_open_agent")

        # Create agent with datetime configuration
        agent_config = AgentConfig(
            agent=AgentInfo(
                name="test_open_agent",
                description="A test open agent for unit testing",
                protocol=ProtocolConfig.RASA,
            ),
            configuration=AgentConfiguration(
                include_date_time=include_date_time,
                timezone=timezone,
            ),
        )
        mcp_open_agent = MCPOpenAgent.from_config(agent_config)

        # Mock get_current_datetime to return a fixed datetime
        mock_now = datetime(2024, 1, 15, 14, 30, 45, tzinfo=ZoneInfo(timezone))
        with patch(
            "rasa.shared.utils.datetime_utils.get_current_datetime"
        ) as mock_get_current_datetime:
            mock_get_current_datetime.return_value = mock_now

            result = mcp_open_agent.render_prompt_template(mock_agent_input)

            if expected_datetime_present:
                # Verify datetime section is present
                assert "### Date & Time Context" in result
                assert expected_date_format in result
                assert expected_time_format in result
                assert expected_day in result
                assert mock_now.tzname() in result
                # Verify get_current_datetime was called
                mock_get_current_datetime.assert_called_once_with(timezone=timezone)
            else:
                # Verify datetime section is NOT present
                assert "### Date & Time Context" not in result
                assert "Current date:" not in result
                assert "Current time:" not in result
                assert "Current day:" not in result
                # Verify get_current_datetime was NOT called
                mock_get_current_datetime.assert_not_called()

    def test_get_task_completed_tool(self):
        """Test getting the task completed tool."""
        tool = MCPOpenAgent.get_task_completed_tool()

        assert tool["type"] == "function"
        assert tool["function"]["name"] == "task_completed"
        assert "FULLY completed" in tool["function"]["description"]
        # task_completed accepts no arguments (empty properties)
        assert tool["function"]["parameters"]["properties"] == {}

    def test_get_agent_specific_built_in_tools(self, mock_agent_input):
        """Test getting agent-specific built-in tools."""
        tools = MCPOpenAgent.get_agent_specific_built_in_tools(mock_agent_input)

        assert len(tools) == 1
        assert tools[0].name == "task_completed"

    @pytest.mark.parametrize(
        "tool_args, expected_message",
        [
            ({"message": "Task completed successfully"}, "Task completed"),
            ({}, "Task completed"),
        ],
    )
    @pytest.mark.asyncio
    async def test_run_task_completed_tool(
        self, mcp_open_agent, mock_agent_input, tool_args, expected_message
    ):
        """Test running the task completed tool with various message configurations.

        The tool always returns result 'Task completed' (tool accepts no arguments).
        """
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="task_completed",
            tool_args=tool_args,
        )

        tool_results = {}
        current_iteration_tool_results = {}

        result = await mcp_open_agent._run_task_completed_tool(
            tool_call,
            mock_agent_input,
            tool_results,
            current_iteration_tool_results,
        )

        assert result.id == mock_agent_input.id
        assert result.status.name == "COMPLETED"
        assert "call_123" in tool_results
        assert tool_results["call_123"].tool_name == "task_completed"
        assert tool_results["call_123"].result == expected_message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "llm_response, expected_status, expected_message_keyword",
        [
            # No/empty response (no choices, no tool_calls) → RECOVERABLE_ERROR
            (
                LLMResponse(
                    id="test_id", created=1642248600, choices=[], tool_calls=None
                ),
                "RECOVERABLE_ERROR",
                "No response from LLM",
            ),
            # Content only with exactly one choice (len(choices)==1) → INPUT_REQUIRED
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
            # Whitespace or empty string → INPUT_REQUIRED, no BotUttered
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=[""],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                None,  # no bot utterance expected
            ),
            (
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["  "],
                    tool_calls=[],
                ),
                "INPUT_REQUIRED",
                None,  # no bot utterance expected
            ),
        ],
    )
    async def test_send_message_llm_response_scenarios(
        self,
        mcp_open_agent,
        mock_agent_input,
        mock_output_channel,
        llm_response,
        expected_status,
        expected_message_keyword,
    ):
        """Test send_message with various LLM response scenarios.

        The content-only path (INPUT_REQUIRED) requires exactly one choice
        (len(llm_response.choices) == 1) and no tool calls.
        Whitespace or empty content yields INPUT_REQUIRED but no BotUttered.
        """
        mock_agent_input.recipient_id = "test_user"
        with patch.object(mcp_open_agent, "llm_client") as mock_llm_client:
            mock_llm_client.acompletion = AsyncMock(return_value=llm_response)

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == expected_status
            if expected_status == "RECOVERABLE_ERROR":
                assert expected_message_keyword in result.error_message
            elif expected_status == "INPUT_REQUIRED":
                assert result.events is not None
                bot_utters = [e for e in result.events if isinstance(e, BotUttered)]
                if expected_message_keyword is None:
                    # Whitespace/empty: no bot utterance
                    assert len(bot_utters) == 0
                else:
                    assert any(
                        expected_message_keyword in (e.text or "") for e in bot_utters
                    )
            else:
                assert result.response_message == expected_message_keyword

    @pytest.mark.asyncio
    async def test_send_message_task_completed_tool(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with task completed tool call."""
        mock_agent_input.recipient_id = "test_user"
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
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=[BotUttered(text="should_not_run")]),
            ) as mock_process_tool_output,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_tool = MagicMock()
            mock_tool.name = "task_completed"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert result.response_message is None
            # Message is in events (BotUttered) as final response
            assert result.events is not None
            bot_texts = [
                e.text for e in result.events if isinstance(e, BotUttered) and e.text
            ]
            assert "Task completed successfully" in bot_texts
            mock_process_tool_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_message_task_completed_processes_regular_tool_results(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Process regular-tool results before returning from task_completed.

        Regression scenario:
        - One LLM response emits multiple tool calls in this order:
          `other_tool`, then `task_completed`.
        - `other_tool` output should still flow through `process_tool_output`
          before the method returns from `task_completed`.
        - The `task_completed` pseudo-tool itself should not be included in
          `current_iteration_tool_results` passed to `process_tool_output`.
        """
        other_call = LLMToolCall(
            id="call_1",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        done_call = LLMToolCall(
            id="call_2",
            type="function",
            tool_name="task_completed",
            tool_args={"message": "done"},
        )
        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Done"],
            tool_calls=[other_call, done_call],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="from_regular_tool")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ) as mock_process_tool_output,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_other = MagicMock()
            mock_other.name = "other_tool"
            mock_done = MagicMock()
            mock_done.name = "task_completed"
            mock_get_tools.return_value = [mock_other, mock_done]
            mock_execute_tool.return_value = tool_output

            mock_agent_input.recipient_id = "test_user"
            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "COMPLETED"
        assert result.response_message is None
        # process_tool_output is called once inside _run_task_completed_tool for
        # current_iteration_tool_results (regular tools only; no task_completed).
        mock_process_tool_output.assert_awaited_once()
        args = mock_process_tool_output.await_args.args
        assert args[0] == {"call_1": tool_output}
        assert args[1]["call_1"] == tool_output
        # Events returned by the hook should survive in final output.
        assert result.events is not None
        assert any(
            isinstance(event, BotUttered) and event.text == "from_regular_tool"
            for event in result.events
        )

    @pytest.mark.asyncio
    async def test_send_message_empty_content_task_completion_retry_stripped_prompt(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Empty content with task_completed triggers retry with stripped system prompt.

        When the LLM returns task_completed but no text content, the agent
        appends a retry system message, strips the original system prompt on the
        next turn, and calls the LLM again with no tools. The second response
        (content only) is treated as the final reply and completes via
        _run_task_completed_tool.
        """
        mock_agent_input.recipient_id = "test_user"
        task_completed_call = LLMToolCall(
            id="call_task_done",
            type="function",
            tool_name="task_completed",
            tool_args={},
        )
        first_response = LLMResponse(
            id="first_id",
            created=1642248600,
            choices=[],  # No content
            tool_calls=[task_completed_call],
        )
        final_text = "Goodbye and take care!"
        second_response = LLMResponse(
            id="second_id",
            created=1642248601,
            choices=[final_text],
            tool_calls=None,
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "task_completed"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.id == mock_agent_input.id
        assert result.status.name == "COMPLETED"
        assert mock_llm_client.acompletion.call_count == 2

        second_call = mock_llm_client.acompletion.call_args_list[1]
        second_messages = second_call.args[0]
        assert (
            second_call.kwargs.get("tools") == []
        ), "Second call must pass no tools after task_completed"

        retry_system_content = "This is the only system instruction for this turn."
        assert second_messages and second_messages[0].get("role") == "system"
        assert retry_system_content in (second_messages[0].get("content") or "")

        assert result.events is not None
        bot_texts = [
            e.text for e in result.events if isinstance(e, BotUttered) and e.text
        ]
        assert final_text in bot_texts

    @pytest.mark.asyncio
    async def test_send_message_filler_message_in_agent_output_events(
        self, mcp_open_agent, mock_agent_input
    ):
        """Agent output events include BotUttered when returning from task_completed.

        When send_message runs with an output_channel and the LLM returns
        task_completed with content, that content is recorded as a BotUttered
        event with type final_response (not filler, since task_completed
        is the only tool call). The final AgentOutput.events must include it.
        """
        mock_agent_input.recipient_id = "user_1"
        mock_agent_input.metadata = {
            AGENT_METADATA_AGENT_ID_KEY: "agent_1",
            AGENT_METADATA_MODEL_ID_KEY: "gpt-4o",
        }
        mock_channel = MagicMock()
        mock_channel.supports_streaming = False
        mock_channel.send_text_message = AsyncMock()

        response_text = "Let me complete that for you."
        mock_tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="task_completed",
            tool_args={"message": "Task completed successfully"},
        )
        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=[response_text],
            tool_calls=[mock_tool_call],
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_tool = MagicMock()
            mock_tool.name = "task_completed"
            mock_get_tools.return_value = [mock_tool]

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_channel
            )

        assert result.id == mock_agent_input.id
        assert result.status.name == "COMPLETED"
        assert result.events is not None
        bot_events = [e for e in result.events if isinstance(e, BotUttered)]
        assert len(bot_events) == 1
        assert bot_events[0].text == response_text
        assert (
            bot_events[0].metadata.get("agent_message_type")
            == BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE
        )

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
                mcp_open_agent, "_create_fatal_error_output"
            ) as mock_fatal_output,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_tool = MagicMock()
            mock_tool.name = tool_name
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = mock_tool_output
            mock_fatal_output.return_value = MagicMock()

            mock_agent_input.recipient_id = "test_user"
            mock_channel = MagicMock()
            mock_channel.supports_streaming = False
            mock_channel.send_text_message = AsyncMock()

            await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_channel
            )

            if expected_behavior == "continue_processing":
                mock_execute_tool.assert_called_once_with(
                    tool_name, tool_args, agent_input=mock_agent_input
                )
            elif expected_behavior == "error_output":
                mock_fatal_output.assert_called_once()
                call_kwargs = mock_fatal_output.call_args[1]
                assert call_kwargs.get("tool_results") is not None

    @pytest.mark.asyncio
    async def test_send_message_malformed_tool_response_retry(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with malformed tool response that triggers retry."""
        mock_agent_input.recipient_id = "test_user"
        mcp_open_agent._include_date_time = False
        # Create a proper exception with original_exception attribute
        decode_error = LLMToolResponseDecodeError("Invalid JSON")
        provider_exception = ProviderClientAPIException("Decode error")
        provider_exception.original_exception = decode_error

        mock_llm_client = MagicMock()
        mock_llm_client.acompletion = AsyncMock(
            side_effect=[
                provider_exception,
                LLMResponse(
                    id="test_id",
                    created=1642248600,
                    choices=["Success response"],
                    tool_calls=[],
                ),
            ]
        )
        mcp_open_agent.llm_client = mock_llm_client

        with (
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(
                mcp_open_agent,
                "build_messages_for_llm_request",
                wraps=mcp_open_agent.build_messages_for_llm_request,
            ) as mock_build_messages,
        ):
            mock_get_tools.return_value = []

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

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
            assert mock_build_messages.call_count == 1

    @pytest.mark.asyncio
    async def test_send_message_general_exception(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message with general exception."""
        mock_agent_input.recipient_id = "test_user"
        with patch.object(mcp_open_agent, "llm_client") as mock_llm_client:
            mock_llm_client.acompletion = AsyncMock(
                side_effect=Exception("General error")
            )

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "FATAL_ERROR"
            assert "I encountered an error" in result.response_message
            assert "General error" in result.error_message

    @pytest.mark.asyncio
    async def test_send_message_max_iterations_reached(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Test send_message when max iterations are reached."""
        mock_agent_input.recipient_id = "test_user"
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
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = mock_tool_output

            # Set max iterations to 1 to force completion
            mcp_open_agent.MAX_ITERATIONS = 1

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

            assert result.id == mock_agent_input.id
            assert result.status.name == "COMPLETED"
            assert "couldn't provide a final answer" in result.response_message

    @pytest.mark.asyncio
    async def test_send_message_calls_process_tool_output_each_iteration(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """process_tool_output is called after each LLM iteration."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        seen_tool_result_sizes = []

        async def _capture_process_tool_output(
            current_iteration_tool_results, cumulative_tool_results, output_channel
        ):
            seen_tool_result_sizes.append(
                (
                    len(current_iteration_tool_results),
                    len(cumulative_tool_results),
                )
            )
            return []

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(side_effect=_capture_process_tool_output),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert seen_tool_result_sizes == [(1, 1)]

    @pytest.mark.asyncio
    async def test_send_message_includes_process_tool_output_events(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Events returned by process_tool_output are added to AgentOutput."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="processed")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert result.events is not None
        processed_events = [
            e
            for e in result.events
            if isinstance(e, BotUttered) and e.text == "processed"
        ]
        assert len(processed_events) == 1

    @pytest.mark.asyncio
    async def test_send_message_exposes_processed_events_to_next_llm_iteration(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Processed tool events are added to subsequent LLM iteration context."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="processed context")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        second_call_messages = mock_llm_client.acompletion.call_args_list[1].args[0]
        assert any(
            message.get("role") == "assistant"
            and message.get("content") == "processed context"
            for message in second_call_messages
        )
        assert any(
            isinstance(event, BotUttered) and event.text == "processed context"
            for event in mock_agent_input.events
        )

    @pytest.mark.asyncio
    async def test_send_message_output_channel_message_needs_bot_event_for_context(
        self, mcp_open_agent, mock_agent_input
    ):
        """A streamed intermediate message is reused when returned as BotUttered."""
        mock_agent_input.recipient_id = "recipient"
        mcp_open_agent._include_date_time = False
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        output_channel = MagicMock()
        output_channel.supports_streaming = False
        output_channel.send_text_message = AsyncMock()

        async def _process_tool_output(
            current_iteration_tool_results, cumulative_tool_results, output_channel_arg
        ):
            await output_channel_arg.send_text_message("recipient", "Processing...")
            return [BotUttered(text="Processing...")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(side_effect=_process_tool_output),
            ),
            patch.object(
                mcp_open_agent,
                "build_messages_for_llm_request",
                wraps=mcp_open_agent.build_messages_for_llm_request,
            ) as mock_build_messages,
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        # send_text_message is also used by the LLM non-streaming path for each
        # response; assert process_tool_output's "Processing..." was sent at
        # least once.
        assert any(
            call.args == ("recipient", "Processing...")
            for call in output_channel.send_text_message.await_args_list
        )
        second_call_messages = mock_llm_client.acompletion.call_args_list[1].args[0]
        assert any(
            message.get("role") == "assistant"
            and message.get("content") == "Processing..."
            for message in second_call_messages
        )
        assert mock_build_messages.call_count == 2

    @pytest.mark.asyncio
    async def test_send_message_rebuilds_context_and_applies_slot_set_events(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Rebuilds per-iteration context and applies SlotSet updates to slots."""
        mock_agent_input.recipient_id = "test_user"
        mcp_open_agent._include_date_time = False
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [SlotSet("user_name", "Alice")]
        original_build_messages = mcp_open_agent.build_messages_for_llm_request
        seen_user_names = []

        def _capture_build_messages(context, turns=10):
            slot_values = {slot.name: slot.value for slot in context.slots}
            seen_user_names.append(slot_values.get("user_name"))
            return original_build_messages(context, turns)

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
            patch.object(
                mcp_open_agent,
                "build_messages_for_llm_request",
                side_effect=_capture_build_messages,
            ) as mock_build_messages,
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert mock_build_messages.call_count == 2
        assert seen_user_names == ["John", "Alice"]
        assert any(
            isinstance(event, SlotSet)
            and event.key == "user_name"
            and event.value == "Alice"
            for event in mock_agent_input.events
        )

    @pytest.mark.asyncio
    async def test_send_message_applies_slot_set_for_unknown_slot(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Unknown SlotSet from process_tool_output is applied in context/output."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Final answer"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [SlotSet("unknown_slot", "value")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "INPUT_REQUIRED"
        assert result.events is not None
        assert any(
            isinstance(event, SlotSet) and event.key == "unknown_slot"
            for event in result.events
        )
        assert any(
            isinstance(event, SlotSet) and event.key == "unknown_slot"
            for event in mock_agent_input.events
        )
        assert any(
            slot.name == "unknown_slot" and slot.type == "any"
            for slot in mock_agent_input.slots
        )

    @pytest.mark.asyncio
    async def test_send_message_task_completed_includes_process_tool_output_events(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Events from process_tool_output survive the task_completed path."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_tool",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        task_completed_call = LLMToolCall(
            id="call_done",
            type="function",
            tool_name="task_completed",
            tool_args={"message": "All done"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Working on it"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Finishing up"],
            tool_calls=[task_completed_call],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )
        process_events = [BotUttered(text="from_hook")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_other = MagicMock()
            mock_other.name = "other_tool"
            mock_completed = MagicMock()
            mock_completed.name = "task_completed"
            mock_get_tools.return_value = [mock_other, mock_completed]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "COMPLETED"
        assert result.events is not None
        assert any(
            isinstance(e, BotUttered) and e.text == "from_hook" for e in result.events
        )

    @pytest.mark.asyncio
    async def test_send_message_tool_error_preserves_processed_events(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Processed tool events are preserved in output when a tool call fails."""
        mock_agent_input.recipient_id = "test_user"
        ok_call = LLMToolCall(
            id="call_ok",
            type="function",
            tool_name="good_tool",
            tool_args={"arg": "v"},
        )
        bad_call = LLMToolCall(
            id="call_bad",
            type="function",
            tool_name="bad_tool",
            tool_args={"arg": "v"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Running tools"],
            tool_calls=[ok_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["More tools"],
            tool_calls=[bad_call],
        )
        ok_output = AgentToolResult(tool_name="good_tool", result="ok", is_error=False)
        bad_output = AgentToolResult(
            tool_name="bad_tool", result=None, is_error=True, error_message="boom"
        )
        process_events = [BotUttered(text="hook_event")]

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(return_value=process_events),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_good = MagicMock()
            mock_good.name = "good_tool"
            mock_bad = MagicMock()
            mock_bad.name = "bad_tool"
            mock_get_tools.return_value = [mock_good, mock_bad]
            mock_execute_tool.side_effect = [ok_output, bad_output]

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "FATAL_ERROR"
        # Optionally verify processed events from the first iteration are preserved
        assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_send_message_process_tool_output_failure_returns_fatal_error(
        self, mcp_open_agent, mock_agent_input, mock_output_channel
    ):
        """Failure in process_tool_output should return FATAL_ERROR."""
        mock_agent_input.recipient_id = "test_user"
        tool_call = LLMToolCall(
            id="call_123",
            type="function",
            tool_name="other_tool",
            tool_args={"arg": "value"},
        )
        first_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Tool executed"],
            tool_calls=[tool_call],
        )
        second_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=[],
        )
        tool_output = AgentToolResult(
            tool_name="other_tool",
            result="Tool result",
            is_error=False,
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
            patch.object(mcp_open_agent, "_execute_tool_call") as mock_execute_tool,
            patch.object(
                mcp_open_agent,
                "process_tool_output",
                new=AsyncMock(side_effect=Exception("hook failure")),
            ),
        ):
            mock_llm_client.acompletion = AsyncMock(
                side_effect=[first_response, second_response]
            )
            mock_tool = MagicMock()
            mock_tool.name = "other_tool"
            mock_get_tools.return_value = [mock_tool]
            mock_execute_tool.return_value = tool_output

            result = await mcp_open_agent.send_message(
                mock_agent_input, output_channel=mock_output_channel
            )

        assert result.status.name == "FATAL_ERROR"
        assert result.error_message is not None
        assert "Failed to process MCP tool output" in result.error_message

    @pytest.mark.asyncio
    async def test_send_message_passes_metadata_to_llm(
        self, mcp_open_agent: MCPOpenAgent, mock_output_channel
    ):
        """Test that send_message correctly passes metadata to
        llm_client.acompletion."""
        # Create agent input with specific metadata and recipient_id
        agent_input = AgentInput(
            id="test_id",
            user_message="Test message",
            slots=[],
            conversation_history="",
            events=[],
            metadata={
                AGENT_METADATA_SENDER_ID_KEY: "user123",
                AGENT_METADATA_AGENT_ID_KEY: "assistant456",
                AGENT_METADATA_MODEL_ID_KEY: "model789",
            },
            timestamp="2024-01-15T10:30:00Z",
        )
        agent_input.recipient_id = "test_user"

        # Create a simple LLM response that will complete immediately (no tool calls)
        mock_llm_response = LLMResponse(
            id="test_id",
            created=1642248600,
            choices=["Test response"],
            tool_calls=[],
        )

        with (
            patch.object(mcp_open_agent, "llm_client") as mock_llm_client,
            patch.object(mcp_open_agent, "get_available_tools") as mock_get_tools,
        ):
            mock_llm_client.acompletion = AsyncMock(return_value=mock_llm_response)
            mock_get_tools.return_value = []

            # Call send_message
            await mcp_open_agent.send_message(
                agent_input, output_channel=mock_output_channel
            )

            # Verify acompletion was called
            assert mock_llm_client.acompletion.called

            # Get the expected metadata
            expected_metadata = mcp_open_agent.get_llm_tracing_metadata(agent_input)

            # Verify the metadata parameter was passed correctly
            call_args = mock_llm_client.acompletion.call_args
            assert call_args is not None
            assert "metadata" in call_args.kwargs
            assert call_args.kwargs["metadata"] == expected_metadata


class TestMCPOpenAgentIsFillerBotUtterance:
    """Tests for :meth:`MCPOpenAgent._is_filler_bot_utterance`."""

    @pytest.mark.parametrize(
        ("tool_names", "expected"),
        [
            (("weather", KEY_TASK_COMPLETED), False),
            ((KEY_TASK_COMPLETED,), False),
            (("weather",), True),
        ],
        ids=[
            "task_completed_with_other_tool",
            "task_completed_only",
            "regular_tool_without_task_completed",
        ],
    )
    def test_is_filler_bot_utterance(
        self,
        mcp_open_agent: MCPOpenAgent,
        tool_names: Tuple[str, ...],
        expected: bool,
    ) -> None:
        llm_response = LLMResponse(
            id="rid",
            created=0,
            choices=["ok"],
            model="m",
            tool_calls=[
                LLMToolCall(id=f"id-{n}", tool_name=n, tool_args={}) for n in tool_names
            ],
        )
        assert (
            mcp_open_agent._is_filler_bot_utterance(
                llm_response, BotUttered(text="Streamed.")
            )
            is expected
        )
