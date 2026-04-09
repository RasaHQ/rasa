import importlib
import json
from typing import Any, Dict, List, Optional

import structlog

from rasa.agents.constants import (
    KEY_CONTENT,
    KEY_ROLE,
    TOOL_ADDITIONAL_PROPERTIES_KEY,
    TOOL_DESCRIPTION_KEY,
    TOOL_NAME_KEY,
    TOOL_PARAMETERS_KEY,
    TOOL_PROPERTIES_KEY,
    TOOL_STRICT_KEY,
    TOOL_TYPE_FUNCTION_KEY,
    TOOL_TYPE_KEY,
)
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import (
    AgentInput,
    AgentOutput,
    AgentToolResult,
    AgentToolSchema,
)
from rasa.core.available_agents import AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel
from rasa.core.constants import (
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE,
    BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY,
)
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.constants import (
    ROLE_SYSTEM,
)
from rasa.shared.core.events import BotUttered, Event
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall

DEFAULT_OPEN_AGENT_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.agents.templates", "mcp_open_agent_prompt_template.jinja2"
)

KEY_TASK_COMPLETED = "task_completed"

TASK_COMPLETED_TOOL = {
    TOOL_TYPE_KEY: TOOL_TYPE_FUNCTION_KEY,
    TOOL_TYPE_FUNCTION_KEY: {
        TOOL_NAME_KEY: KEY_TASK_COMPLETED,
        TOOL_DESCRIPTION_KEY: (
            "Call this tool exactly once when your primary task is FULLY completed. "
            "This tool accepts no arguments. You MUST also include text in the same "
            "response: a natural, conversational follow-up to the user's last message "
            "(e.g. acknowledge their choice, wish them well, close warmly). Do NOT "
            "summarize what you did or what happened in the conversation. Do NOT "
            "include any inner thoughts or explanations. A response with only the "
            "tool call and no text is invalid. Keep it short and natural."
        ),
        TOOL_PARAMETERS_KEY: {
            TOOL_TYPE_KEY: "object",
            TOOL_PROPERTIES_KEY: {},
            TOOL_ADDITIONAL_PROPERTIES_KEY: False,
        },
        TOOL_STRICT_KEY: True,
    },
}

structlogger = structlog.get_logger()


class MCPOpenAgent(MCPBaseAgent):
    """MCP protocol implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        protocol_type: ProtocolConfig,
        server_configs: List[AgentMCPServerConfig],
        llm_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        include_date_time: Optional[bool] = None,
        timezone: Optional[str] = None,
        enable_filler_messages: Optional[bool] = None,
        tool_timeout: Optional[float] = None,
    ):
        super().__init__(
            name,
            description,
            protocol_type,
            server_configs,
            llm_config,
            prompt_template,
            timeout,
            max_retries,
            include_date_time,
            timezone,
            enable_filler_messages,
            tool_timeout,
        )

    @property
    def protocol_type(self) -> ProtocolType:
        return ProtocolType.MCP_OPEN

    @staticmethod
    def get_default_prompt_template() -> str:
        return DEFAULT_OPEN_AGENT_PROMPT_TEMPLATE

    @staticmethod
    def get_task_completed_tool() -> Dict[str, Any]:
        """Get the task completed tool for MCP. Override to customize/disable."""
        return TASK_COMPLETED_TOOL

    @classmethod
    def get_agent_specific_built_in_tools(
        cls, agent_input: AgentInput
    ) -> List[AgentToolSchema]:
        """Get agentic specific built-in tools."""
        return [AgentToolSchema.from_litellm_json_format(cls.get_task_completed_tool())]

    @staticmethod
    def get_system_message_for_empty_content_at_task_completion() -> Dict[str, str]:
        """Get the system message for an empty content with task completed tool call."""
        system_message = (
            "This is the only system instruction for this turn. The goal is already "
            "completed and the agent will stop after your response. Do not call any "
            "tools. Generate only the final response to the user: a natural, "
            "conversational follow-up to the user's last message (e.g. acknowledge "
            "their choice, wish them well, close warmly). Do NOT summarize what you "
            "did or what happened in the conversation. Do NOT include any inner "
            "thoughts or explanations. Keep it short and natural."
        )
        return {
            KEY_ROLE: ROLE_SYSTEM,
            KEY_CONTENT: system_message,
        }

    def _append_empty_content_at_task_completion_system_message(
        self, messages: List[Dict[str, Any]]
    ) -> None:
        """Log empty content at task completion and append the retry system message."""
        structlogger.debug(
            "mcp_open_agent.send_message.empty_content_at_task_completion",
            event_info=(
                "Empty content with task completed tool call "
                "received from LLM. Retrying the LLM call."
            ),
            agent_name=self._name,
            agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
        )
        messages.append(self.get_system_message_for_empty_content_at_task_completion())

    async def _run_task_completed_tool(
        self,
        tool_call: Optional[LLMToolCall],
        agent_input: AgentInput,
        tool_results: Dict[str, AgentToolResult],
        current_iteration_tool_results: Dict[str, AgentToolResult],
        generated_events: Optional[List[Event]] = None,
        accumulated_tool_output_events: Optional[List[Event]] = None,
        output_channel: Optional[OutputChannel] = None,
        bot_uttered: Optional[BotUttered] = None,
    ) -> AgentOutput:
        """Run the task completed tool."""
        generated_events = generated_events or []
        accumulated_tool_output_events = accumulated_tool_output_events or []
        # A single response can include regular tools before
        # `task_completed`. Process those pending results first so
        # they still go through `process_tool_output` for this
        # iteration. `task_completed` itself is excluded.
        if current_iteration_tool_results:
            events_from_tool_results = await self._process_tool_output_or_raise(
                current_iteration_tool_results,
                tool_results,
                output_channel,
            )
            if events_from_tool_results:
                self._apply_slot_set_events_to_agent_input(
                    agent_input, events_from_tool_results
                )
                accumulated_tool_output_events.extend(events_from_tool_results)
                agent_input.events.extend(events_from_tool_results)

        if tool_call:
            tool_result = AgentToolResult(
                tool_name=tool_call.tool_name,
                result="Task completed",
            )
            tool_results[tool_call.id] = tool_result

        # Record the final response as a BotUttered event.
        if bot_uttered:
            bot_uttered.metadata[BOT_UTTERANCE_AGENT_MESSAGE_TYPE_KEY] = (
                BOT_UTTERANCE_AGENT_MESSAGE_TYPE_FINAL_RESPONSE
            )
            generated_events.append(bot_uttered)

        # Create the agent output for the task completed tool.
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message=None,
            events=generated_events + accumulated_tool_output_events,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results
            ),
        )

    def _is_task_completed_tool_call_present(self, llm_response: LLMResponse) -> bool:
        """Check if the task completed tool call is present in the LLM response."""
        if llm_response.tool_calls and any(
            tool_call.tool_name == KEY_TASK_COMPLETED
            for tool_call in llm_response.tool_calls
        ):
            return True
        return False

    def _is_filler_bot_utterance(
        self,
        llm_response: LLMResponse,
        bot_uttered: Optional[BotUttered],
    ) -> bool:
        """Whether streamed assistant text should be recorded as a filler message."""
        return (
            bot_uttered is not None
            and self._enable_filler_messages
            and bool(llm_response.tool_calls)
            and not self._is_task_completed_tool_call_present(llm_response)
        )

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the LLM and return the response.

        Each iteration generates and sends the LLM response via
        ``generate_and_send_response`` and routes:
        - no content and no tool calls              → RECOVERABLE_ERROR
        - content only                              → content already streamed;
                                                      return INPUT_REQUIRED
        - task_completed with no content            → append retry system message
                                                      and loop again
        - task_completed with content / other tools → execute tools; task-exit tools
                                                      break the loop

        Any content streamed alongside tool calls is recorded as a BotUttered event
        in ``generated_events`` so the conversation history stays consistent.
        """
        _available_tools = self.get_available_tools(agent_input)
        _available_tools_names = [tool.name for tool in _available_tools]
        message_build_cache: Dict[str, Any] = {}
        tool_call_messages: List[Dict[str, Any]] = []
        tool_results: Dict[str, AgentToolResult] = {}
        generated_events: List[Event] = []
        # Stores events returned by `process_tool_output`, accumulated across
        # all completed iterations in this `send_message` run.
        accumulated_tool_output_events: List[Event] = []

        # Convert available tools to OpenAI JSON format
        tools_in_openai_format = [
            tool.to_litellm_json_format() for tool in _available_tools
        ]

        task_completed: bool = False

        for iteration in range(self.MAX_ITERATIONS):
            current_iteration_tool_results: Dict[str, AgentToolResult] = {}
            try:
                messages = self._build_messages_for_llm_request_with_cache(
                    agent_input,
                    message_build_cache,
                    strip_original_system_prompt=task_completed,
                )
                messages.extend(tool_call_messages)

                if task_completed:
                    # Move system message(s) to the front
                    system_messages = [
                        m for m in messages if m.get(KEY_ROLE) == ROLE_SYSTEM
                    ]
                    other_messages = [
                        m for m in messages if m.get(KEY_ROLE) != ROLE_SYSTEM
                    ]
                    messages = system_messages + other_messages

                structlogger.debug(
                    "mcp_open_agent.send_message.iteration",
                    event_info=(
                        f"Starting iteration {iteration + 1} for agent {self._name}"
                    ),
                    agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
                    highlight=True,
                )
                # Make the LLM call using the llm_client
                structlogger.debug(
                    "mcp_open_agent.send_message.sending_message_to_llm",
                    messages=messages,
                    json_formatting=["messages"],
                    event_info=f"Sending message to LLM (iteration {iteration + 1})",
                    agent_name=self._name,
                    agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
                )

                llm_response, bot_uttered = await self.generate_and_send_response(
                    messages=messages,
                    tools=tools_in_openai_format if not task_completed else [],
                    metadata=self.get_llm_tracing_metadata(agent_input),
                    agent_input=agent_input,
                    output_channel=output_channel,
                )

                llm_content = (
                    llm_response.choices[0]
                    if llm_response and llm_response.choices
                    else None
                )
                # If no response from LLM, return an error output.
                if llm_response is None or not (
                    llm_response.choices or llm_response.tool_calls
                ):
                    return self._create_recoverable_error_output(
                        agent_input,
                        "No response from LLM.",
                        generated_events=generated_events
                        + accumulated_tool_output_events,
                        tool_results=tool_results,
                        log_event_name="mcp_open_agent.send_message.no_llm_response",
                    )

                # Continue the loop if the LLM response is empty and the task completed
                # tool call is present.
                if (
                    not llm_content
                    and llm_response.tool_calls
                    and self._is_task_completed_tool_call_present(llm_response)
                ):
                    self._append_empty_content_at_task_completion_system_message(
                        tool_call_messages
                    )
                    task_completed = True
                    continue

                # Content only (no tool calls) → content already streamed
                # return INPUT_REQUIRED if task is not yet completed yet
                # otherwise, return COMPLETED
                if not llm_response.tool_calls and len(llm_response.choices) == 1:
                    if task_completed:
                        return await self._run_task_completed_tool(
                            tool_call=None,
                            agent_input=agent_input,
                            tool_results=tool_results,
                            current_iteration_tool_results=current_iteration_tool_results,
                            generated_events=generated_events,
                            accumulated_tool_output_events=accumulated_tool_output_events,
                            output_channel=output_channel,
                            bot_uttered=bot_uttered,
                        )

                    # Record the content as a BotUttered event.
                    self._record_input_required_bot_uttered(
                        bot_uttered, generated_events
                    )
                    return self._create_input_required_output(
                        agent_input,
                        llm_content,
                        output_channel,
                        generated_events + accumulated_tool_output_events,
                        tool_results,
                    )

                # Record the filler message if it is present
                is_filler = self._is_filler_bot_utterance(llm_response, bot_uttered)
                if output_channel:
                    output_channel.note_last_streamed_bot_message_was_filler(is_filler)

                if is_filler:
                    # Record the filler message as a BotUttered event.
                    self._record_filler_bot_uttered(bot_uttered, generated_events)

                if llm_response.tool_calls:
                    # Add the assistant message with tool calls to the messages.
                    tool_call_messages.append(
                        self._get_assistant_message_with_tool_calls(llm_response)
                    )

                    for tool_call in llm_response.tool_calls:
                        structlogger.debug(
                            "mcp_open_agent.send_message.tool_call",
                            event_info=f"Processing tool call {tool_call.tool_name}",
                            tool_name=tool_call.tool_name,
                            tool_args=json.dumps(tool_call.tool_args),
                            agent_name=self._name,
                            agent_id=str(
                                make_agent_identifier(self._name, self.protocol_type)
                            ),
                            json_formatting=["tool_args"],
                        )

                        # If the tool is not available, return a fatal error output.
                        if tool_call.tool_name not in _available_tools_names:
                            return self._create_fatal_error_output(
                                agent_input,
                                f"Tool {tool_call.tool_name} is not available.",
                                "mcp_open_agent.send_message.tool_not_available",
                                events=generated_events
                                + accumulated_tool_output_events,
                                tool_results=tool_results,
                                tool_name=tool_call.tool_name,
                            )

                        # task_completed → exit immediately
                        if tool_call.tool_name == KEY_TASK_COMPLETED:
                            return await self._run_task_completed_tool(
                                tool_call,
                                agent_input,
                                tool_results,
                                current_iteration_tool_results,
                                generated_events=generated_events,
                                accumulated_tool_output_events=accumulated_tool_output_events,
                                output_channel=output_channel,
                                bot_uttered=bot_uttered,
                            )

                        # All other tools → execute and continue the loop
                        if error_output := await self._process_tool_call(
                            tool_call,
                            agent_input,
                            tool_call_messages,
                            tool_results,
                            current_iteration_tool_results,
                            "mcp_open_agent.send_message.tool_output",
                            events=generated_events + accumulated_tool_output_events,
                        ):
                            return error_output

                    mcp_tool_events = self._get_mcp_tool_executed_events(
                        llm_response.tool_calls, current_iteration_tool_results
                    )
                    accumulated_tool_output_events.extend(mcp_tool_events)
                    if mcp_tool_events:
                        agent_input.events.extend(mcp_tool_events)

                    events_from_tool_results = await self._process_tool_output_or_raise(
                        current_iteration_tool_results,
                        tool_results,
                        output_channel,
                    )
                    if events_from_tool_results:
                        self._apply_slot_set_events_to_agent_input(
                            agent_input, events_from_tool_results
                        )
                        accumulated_tool_output_events.extend(events_from_tool_results)
                        agent_input.events.extend(events_from_tool_results)

            except Exception as e:
                if self._is_malformed_tool_response_exception(e):
                    # Continue to make another LLM call by breaking out of the current
                    # iteration and letting the loop continue with a fresh LLM request
                    self._append_malformed_tool_response_system_message(
                        tool_call_messages, agent_input, e, "mcp_open_agent"
                    )
                    continue
                return self._create_fatal_error_output(
                    agent_input,
                    str(e),
                    "mcp_open_agent.send_message.error_in_agent_loop",
                    events=generated_events + accumulated_tool_output_events,
                    tool_results=tool_results,
                )
        return self._create_max_iterations_reached_output(
            agent_input,
            events=generated_events + accumulated_tool_output_events,
            tool_results=tool_results,
        )
