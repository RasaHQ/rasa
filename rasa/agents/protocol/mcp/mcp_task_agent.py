import importlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import structlog
from jinja2 import Template

from rasa.agents.constants import (
    KEY_CONTENT,
    KEY_ROLE,
    KEY_TOOL_CALL_ID,
)
from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.protocol.mcp.mcp_base_agent import MCPBaseAgent
from rasa.agents.schemas import (
    AgentInput,
    AgentOutput,
    AgentToolResult,
    AgentToolSchema,
)
from rasa.agents.schemas.agent_input import AgentInputSlot
from rasa.agents.utils import get_slot_value_from_agent_input
from rasa.core.available_agents import AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.constants import (
    ROLE_TOOL,
)
from rasa.shared.core.constants import MOCKED_DATETIME_SLOT
from rasa.shared.core.events import SlotSet
from rasa.shared.exceptions import (
    LLMToolResponseDecodeError,
    ProviderClientAPIException,
)
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.datetime_utils import resolve_datetime
from rasa.utils.pypred import Predicate

DEFAULT_TASK_AGENT_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.agents.templates", "mcp_task_agent_prompt_template.jinja2"
)

structlogger = structlog.get_logger()


class MCPTaskAgent(MCPBaseAgent):
    """MCPTaskAgent client implementation"""

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
        return ProtocolType.MCP_TASK

    @staticmethod
    def get_default_prompt_template() -> str:
        return DEFAULT_TASK_AGENT_PROMPT_TEMPLATE

    @classmethod
    def get_agent_specific_built_in_tools(
        cls, agent_input: AgentInput
    ) -> List[AgentToolSchema]:
        """Get agentic specific built-in tools."""
        slot_names = cls._get_slot_names_from_exit_conditions(agent_input)
        slot_definitions = [
            slot for slot in agent_input.slots if slot.name in slot_names
        ]

        return [
            AgentToolSchema.from_litellm_json_format(
                cls.get_slot_specific_set_slot_tool(slot)
            )
            for slot in slot_definitions
        ]

    @classmethod
    def _get_slot_names_from_exit_conditions(cls, agent_input: AgentInput) -> List[str]:
        """Extract valid slot names from exit conditions."""
        exit_conditions = agent_input.metadata.get("exit_if", [])

        # Find all unique names matching "slots.<name>"
        extracted_slot_names = {
            name
            for condition in exit_conditions
            for name in re.findall(r"\bslots\.(\w+)", condition)
        }

        slot_names = [slot.name for slot in agent_input.slots]

        # Keep only slots that actually exist in agent_input.slots
        valid_slot_names = [
            slot_name for slot_name in extracted_slot_names if slot_name in slot_names
        ]

        return valid_slot_names

    @classmethod
    def get_slot_specific_set_slot_tool(cls, slot: AgentInputSlot) -> Dict[str, Any]:
        """Get the set slot tool."""
        tool_description = f"Set the slot '{slot.name}' to a specific value. "
        tool_description += f"The slot type is {slot.type}."
        if slot.type == "categorical":
            tool_description += f" The allowed values are: {slot.allowed_values}."

        return {
            "type": "function",
            "function": {
                "name": f"set_slot_{slot.name}",
                "description": tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot_value": {
                            "type": "string",
                            "description": "The value to assign to the slot.",
                        },
                    },
                    "required": ["slot_value"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    @classmethod
    def _is_exit_conditions_met(
        cls, agent_input: AgentInput, slots: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if the exit conditions are met.

        Args:
            agent_input: The agent input.
            slots: The slots to check the exit conditions against.

        Returns:
            A tuple containing a boolean indicating if the exit conditions are met
            and a string indicating if an internal error occurred.
        """
        if not slots:
            return False, None

        exit_conditions = agent_input.metadata.get("exit_if", [])
        current_context = {"slots": slots}

        internal_error = None
        all_conditions_met = True

        for condition in exit_conditions:
            try:
                rendered_template = Template(condition).render(current_context)
                predicate = Predicate(rendered_template)
                condition_result = predicate.evaluate(current_context)

                # All conditions must be met (AND logic)
                if not condition_result:
                    all_conditions_met = False
                    break

            except (TypeError, Exception) as e:
                structlogger.error(
                    "mcp_task_agent.is_exit_conditions_met.predicate.error",
                    predicate=condition,
                    error=str(e),
                )
                all_conditions_met = False
                internal_error = str(e)
                break

        if internal_error:
            structlogger.debug(
                "mcp_task_agent.is_exit_conditions_met.result",
                event_info="Failed to evaluate exit conditions - error occurred",
                exit_conditions=exit_conditions,
                error=internal_error,
            )
        else:
            structlogger.debug(
                "mcp_task_agent.is_exit_conditions_met.result",
                event_info=f"Exit conditions met: {all_conditions_met}",
                evaulation_result=all_conditions_met,
                exit_conditions=exit_conditions,
            )

        return all_conditions_met, internal_error

    def _get_slot_name_from_tool_name(self, tool_name: str) -> Optional[str]:
        """Get the slot name from the tool name."""
        match = re.match(r"^set_slot_(\w+)$", tool_name)
        if match:
            return match.group(1)
        return None

    def _run_set_slot_tool(
        self, slot_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the set slot tool."""
        slot_value = arguments.get("slot_value")

        # Handle type conversion for common cases
        if isinstance(slot_value, str):
            # Convert common boolean strings to actual booleans
            if slot_value.lower() == "true":
                slot_value = True
            elif slot_value.lower() == "false":
                slot_value = False

        return {slot_name: slot_value}

    def _generate_agent_task_completed_output(
        self,
        agent_input: AgentInput,
        slots: Dict[str, Any],
        tool_results: Dict[str, AgentToolResult],
    ) -> AgentOutput:
        """Generate an agent task completed output."""
        _slot_names_to_be_filled = self._get_slot_names_from_exit_conditions(
            agent_input
        )
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            events=[
                SlotSet(slot_name, slot_value)
                for slot_name, slot_value in slots.items()
                if slot_name in _slot_names_to_be_filled
            ],
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results
            ),
        )

    def render_prompt_template(self, context: AgentInput) -> str:
        """Render the prompt template with the provided inputs."""
        slot_names = self._get_slot_names_from_exit_conditions(context)

        template_vars = {
            **context.model_dump(exclude={"id", "timestamp", "events"}),
            "description": self._description,
            "slot_names": slot_names,
        }

        # Add current_datetime object if enabled
        if self._include_date_time:
            mocked_datetime_value = get_slot_value_from_agent_input(
                context, MOCKED_DATETIME_SLOT
            )
            template_vars["current_datetime"] = resolve_datetime(
                mocked_datetime_value, timezone=self._timezone
            )
        return Template(self.prompt_template).render(**template_vars)

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the LLM and return the response."""
        messages = self.build_messages_for_llm_request(agent_input)
        tool_results: Dict[str, AgentToolResult] = {}

        _slot_values = {slot.name: slot.value for slot in agent_input.slots}
        _available_tools = self.get_available_tools(agent_input)
        _available_tools_names = [tool.name for tool in _available_tools]

        # Convert available tools to OpenAI JSON format
        tools_in_openai_format = [
            tool.to_litellm_json_format() for tool in _available_tools
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
                    "mcp_task_agent.send_message.sending_message_to_llm",
                    messages=messages,
                    json_formatting=["messages"],
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
                        "mcp_task_agent.send_message.no_llm_response",
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
                            "mcp_task_agent.send_message.tool_call",
                            event_info=f"Processing tool call {tool_call.tool_name}",
                            tool_name=tool_call.tool_name,
                            tool_args=json.dumps(tool_call.tool_args),
                            json_formatting=["tool_args"],
                            agent_name=self._name,
                            agent_id=str(
                                make_agent_identifier(self._name, self.protocol_type)
                            ),
                        )

                        # If the tool is not available, return a fatal error output.
                        if tool_call.tool_name not in _available_tools_names:
                            event_info = f"Tool {tool_call.tool_name} is not available."
                            structlogger.error(
                                "mcp_task_agent.send_message.tool_not_available",
                                tool_name=tool_call.tool_name,
                                event_info=event_info,
                                user_message=agent_input.user_message,
                            )
                            return AgentOutput(
                                id=agent_input.id,
                                status=AgentStatus.FATAL_ERROR,
                                error_message=event_info,
                            )

                        if slot_name := self._get_slot_name_from_tool_name(
                            tool_call.tool_name
                        ):
                            if (
                                slot_name in _slot_values
                                and "slot_value" in tool_call.tool_args
                            ):
                                _slot_values.update(
                                    self._run_set_slot_tool(
                                        slot_name, tool_call.tool_args
                                    )
                                )

                                # Add the tool call message to the messages for
                                # slot-setting tools
                                messages.append(
                                    {
                                        KEY_ROLE: ROLE_TOOL,
                                        KEY_TOOL_CALL_ID: tool_call.id,
                                        KEY_CONTENT: f"Slot {slot_name} set to "
                                        f"{tool_call.tool_args.get('slot_value')}",
                                    }
                                )
                            else:
                                return AgentOutput(
                                    id=agent_input.id,
                                    status=AgentStatus.FATAL_ERROR,
                                    error_message=(
                                        f"The slot `{slot_name}` that the tool "
                                        f"`{tool_call.tool_name}` is trying to set "
                                        f"is not found in agent input."
                                    ),
                                )
                        else:
                            # Execute the tool call.
                            tool_output = await self._execute_tool_call(
                                tool_call.tool_name,
                                tool_call.tool_args,
                            )

                            structlogger.debug(
                                "mcp_task_agent.send_message.tool_output",
                                event_info=(
                                    f"Tool output for tool call {tool_call.tool_name}"
                                ),
                                tool_output=tool_output.model_dump(),
                                json_formatting=["tool_output"],
                                tool_name=tool_call.tool_name,
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

                        exit_met, internal_error = self._is_exit_conditions_met(
                            agent_input, _slot_values
                        )

                        # Agent signals task completion if exit conditions are met.
                        if exit_met:
                            return self._generate_agent_task_completed_output(
                                agent_input, _slot_values, tool_results
                            )

                        # If an internal error occurred while checking the exit
                        # conditions, return a fatal error output.
                        if internal_error:
                            return AgentOutput(
                                id=agent_input.id,
                                status=AgentStatus.FATAL_ERROR,
                                response_message=(
                                    "An internal error occurred while checking the "
                                    "exit conditions."
                                ),
                                structured_results=self._get_structured_results_for_agent_output(
                                    agent_input, tool_results
                                ),
                                error_message=internal_error,
                            )

            except Exception as e:
                if isinstance(e, ProviderClientAPIException) and isinstance(
                    e.original_exception, LLMToolResponseDecodeError
                ):
                    structlogger.debug(
                        "mcp_task_agent.send_message.malformed_tool_response_error",
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
                    "mcp_task_agent.send_message.error_in_agent_loop",
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
