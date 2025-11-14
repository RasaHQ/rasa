import importlib
import json
from typing import Any, Dict, List, Optional

import structlog

from rasa.agents.constants import (
    KEY_CONTENT,
    KEY_ROLE,
    KEY_TOOL_CALL_ID,
    TOOL_ADDITIONAL_PROPERTIES_KEY,
    TOOL_DESCRIPTION_KEY,
    TOOL_NAME_KEY,
    TOOL_PARAMETERS_KEY,
    TOOL_PROPERTIES_KEY,
    TOOL_REQUIRED_KEY,
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
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.constants import (
    ROLE_TOOL,
)
from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm.llm_response import LLMResponse, LLMToolCall

DEFAULT_OPEN_AGENT_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.agents.templates", "mcp_open_agent_prompt_template.jinja2"
)

KEY_TASK_COMPLETED = "task_completed"

TASK_COMPLETED_TOOL = {
    TOOL_TYPE_KEY: TOOL_TYPE_FUNCTION_KEY,
    TOOL_TYPE_FUNCTION_KEY: {
        TOOL_NAME_KEY: KEY_TASK_COMPLETED,
        TOOL_DESCRIPTION_KEY: "Signal that the MCP agent has FULLY completed its "
        "primary task. Once you have presented your findings, follow-up with "
        "a message summarizing the completed task in a comprehensive and well-written "
        "manner. Avoid repeating information already provided in the conversation.",
        TOOL_PARAMETERS_KEY: {
            TOOL_TYPE_KEY: "object",
            TOOL_PROPERTIES_KEY: {
                "message": {
                    TOOL_TYPE_KEY: "string",
                    TOOL_DESCRIPTION_KEY: "A message describing the completed task.",
                }
            },
            TOOL_REQUIRED_KEY: ["message"],
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

    def _run_task_completed_tool(
        self,
        tool_call: LLMToolCall,
        agent_input: AgentInput,
        tool_results: Dict[str, AgentToolResult],
    ) -> AgentOutput:
        """Run the task completed tool."""
        # Create the agent tool result for the task completed tool.

        tool_result = AgentToolResult(
            tool_name=tool_call.tool_name,
            result=tool_call.tool_args.get("message", "Task completed"),
        )
        tool_results[tool_call.id] = tool_result

        # Create the agent output for the task completed tool.
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message=tool_result.result,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results
            ),
        )

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the LLM and return the response."""
        messages = self.build_messages_for_llm_request(agent_input)
        tool_results: Dict[str, AgentToolResult] = {}
        # Convert available tools to OpenAI JSON format
        tools_in_openai_format = [
            tool.to_litellm_json_format()
            for tool in self.get_available_tools(agent_input)
        ]

        for iteration in range(self.MAX_ITERATIONS):
            try:
                structlogger.debug(
                    "mcp_task_agent.send_message.iteration",
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
                    event_info=f"Sending message to LLM (iteration {iteration + 1})",
                    agent_name=self._name,
                    agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
                )
                llm_response = LLMResponse.ensure_llm_response(
                    await self.llm_client.acompletion(
                        messages,
                        tools=tools_in_openai_format,
                        metadata=self.get_llm_tracing_metadata(agent_input),
                    )
                )

                # If no response from LLM, return an error output.
                if llm_response is None or not (
                    llm_response.choices or llm_response.tool_calls
                ):
                    event_info = "No response from LLM."
                    structlogger.warning(
                        "mcp_open_agent.send_message.no_llm_response",
                        event_info=event_info,
                        agent_name=self._name,
                        agent_id=str(
                            make_agent_identifier(self._name, self.protocol_type)
                        ),
                    )
                    return AgentOutput(
                        id=agent_input.id,
                        status=AgentStatus.RECOVERABLE_ERROR,
                        error_message=event_info,
                        structured_results=(
                            self._get_structured_results_for_agent_output(
                                agent_input, tool_results
                            )
                        ),
                    )

                # If no tool calls, return the response directly with input required.
                if not llm_response.tool_calls and len(llm_response.choices) == 1:
                    return AgentOutput(
                        id=agent_input.id,
                        status=AgentStatus.INPUT_REQUIRED,
                        response_message=llm_response.choices[0],
                        structured_results=(
                            self._get_structured_results_for_agent_output(
                                agent_input, tool_results
                            )
                        ),
                    )

                # If there are tool calls, process them.
                if llm_response.tool_calls:
                    # Add the assistant message with tool calls to the messages.
                    messages.append(
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

                        # Agent signals task completion.
                        if tool_call.tool_name == KEY_TASK_COMPLETED:
                            return self._run_task_completed_tool(
                                tool_call, agent_input, tool_results
                            )

                        else:
                            # Execute the tool call.
                            tool_output = await self._execute_tool_call(
                                tool_call.tool_name,
                                tool_call.tool_args,
                            )

                            structlogger.debug(
                                "mcp_open_agent.send_message.tool_output",
                                event_info=(
                                    f"Tool output for tool call {tool_call.tool_name}"
                                ),
                                tool_output=tool_output.model_dump(),
                                json_formatting=["tool_output"],
                                agent_name=self._name,
                                agent_id=str(
                                    make_agent_identifier(
                                        self._name, self.protocol_type
                                    )
                                ),
                            )

                            # If the tool call failed, generate an agent error output.
                            if tool_output.is_error or tool_output.result is None:
                                return self._generate_agent_error_output(
                                    tool_output, agent_input, tool_call
                                )

                            # Store the tool output in the tool_results.
                            tool_results[tool_call.id] = tool_output

                            # Add the tool call message to the messages.
                            messages.append(
                                {
                                    KEY_ROLE: ROLE_TOOL,
                                    KEY_TOOL_CALL_ID: tool_call.id,
                                    KEY_CONTENT: tool_output.result,
                                }
                            )

            except Exception as e:
                if isinstance(e, ProviderClientAPIException) and isinstance(
                    e.original_exception, LLMToolResponseDecodeError
                ):
                    structlogger.debug(
                        "mcp_open_agent.send_message.malformed_tool_response_error",
                        event_info=(
                            "Malformed tool response received from LLM "
                            "(JSON decode error). Retrying the LLM call."
                        ),
                        user_message=agent_input.user_message,
                        agent_name=self._name,
                        agent_id=str(
                            make_agent_identifier(self._name, self.protocol_type)
                        ),
                        original_exception=str(e.original_exception),
                    )
                    # Continue to make another LLM call by breaking out of the current
                    # iteration and letting the loop continue with a fresh LLM request
                    messages.append(
                        self._get_system_message_for_malformed_tool_response()
                    )
                    continue
                structlogger.error(
                    "mcp_open_agent.send_message.error_in_agent_loop",
                    event_info=f"Failed to send message: {e}",
                    user_message=agent_input.user_message,
                    agent_name=self._name,
                    agent_id=str(make_agent_identifier(self._name, self.protocol_type)),
                )
                return AgentOutput(
                    id=agent_input.id,
                    status=AgentStatus.FATAL_ERROR,
                    response_message=f"I encountered an error: {e!s}",
                    structured_results=self._get_structured_results_for_agent_output(
                        agent_input, tool_results
                    ),
                    error_message=str(e),
                )
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message=(
                "I've completed my research but couldn't provide a final answer within"
                "the allowed steps."
            ),
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results
            ),
        )
