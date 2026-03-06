import importlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple, cast

import structlog
from jinja2 import Template

from rasa.agents.constants import (
    AGENT_METADATA_EXIT_IF_KEY,
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
from rasa.core.available_agents import AgentMCPServerConfig, ProtocolConfig
from rasa.core.channels import OutputChannel
from rasa.shared.agents.utils import make_agent_identifier
from rasa.shared.constants import (
    ROLE_TOOL,
)
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.providers.llm.llm_response import LLMToolCall
from rasa.utils.pypred import Predicate

DEFAULT_TASK_AGENT_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.agents.templates", "mcp_task_agent_prompt_template.jinja2"
)

structlogger = structlog.get_logger()


class MCPTaskAgent(MCPBaseAgent):
    """MCPTaskAgent client implementation."""

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
        exit_conditions = agent_input.metadata.get(AGENT_METADATA_EXIT_IF_KEY, [])

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

        exit_conditions = agent_input.metadata.get(AGENT_METADATA_EXIT_IF_KEY, [])
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
                evaluation_result=all_conditions_met,
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

    def _handle_slot_setting_tool(
        self,
        agent_input: AgentInput,
        slot_name: str,
        tool_call: LLMToolCall,
        current_slot_values: Dict[str, Any],
        initial_slot_values: Dict[str, Any],
        tool_call_messages: List[Dict[str, Any]],
        generated_events: List[Event],
        accumulated_tool_output_events: List[Event],
        tool_results: Dict[str, AgentToolResult],
    ) -> Optional[AgentOutput]:
        """Apply slot-setting tool or return fatal error if slot not found/invalid.

        Returns None if the slot was applied successfully, otherwise returns
        an AgentOutput for the fatal error.
        """
        if slot_name in current_slot_values and "slot_value" in tool_call.tool_args:
            current_slot_values.update(
                self._run_set_slot_tool(slot_name, tool_call.tool_args)
            )
            # Add the tool call message to the messages for slot-setting tools
            tool_call_messages.append(
                {
                    KEY_ROLE: ROLE_TOOL,
                    KEY_TOOL_CALL_ID: tool_call.id,
                    KEY_CONTENT: f"Slot {slot_name} set to "
                    f"{tool_call.tool_args.get('slot_value')}",
                }
            )
            return None
        return self._create_fatal_error_output(
            agent_input,
            (
                f"The slot `{slot_name}` that the tool "
                f"`{tool_call.tool_name}` is trying to set "
                f"is not found in agent input."
            ),
            "mcp_task_agent.send_message.slot_not_found",
            events=self.get_events_for_agent_output(
                agent_input,
                initial_slot_values,
                current_slot_values,
                generated_events + accumulated_tool_output_events,
            ),
            tool_results=tool_results,
        )

    def _get_slot_set_events_for_changed_slots(
        self,
        agent_input: AgentInput,
        initial_slot_values: Dict[str, Any],
        current_slot_values: Dict[str, Any],
    ) -> List[SlotSet]:
        """Return SlotSet events for exit-condition slots that changed."""
        slot_names_to_be_filled = self._get_slot_names_from_exit_conditions(agent_input)
        return [
            SlotSet(slot_name, current_slot_values[slot_name])
            for slot_name in slot_names_to_be_filled
            if slot_name in current_slot_values
            and current_slot_values[slot_name] != initial_slot_values.get(slot_name)
        ]

    def get_events_for_agent_output(
        self,
        agent_input: AgentInput,
        initial_slot_values: Dict[str, Any],
        current_slot_values: Dict[str, Any],
        generated_events: List[Event],
    ) -> List[Event]:
        """Return the full event list to attach to an AgentOutput.

        Combines SlotSet events for exit-condition slots that changed value
        during the agent loop with any ``BotUttered`` (or other) events
        accumulated in ``generated_events``.

        Args:
            agent_input: The original agent input for this turn.
            initial_slot_values: Slot values at the start of the agent loop.
            current_slot_values: Slot values after tool execution.
            generated_events: Events accumulated during the agent loop
                (e.g. BotUttered events for streamed content).

        Returns:
            A combined list of generated events followed by SlotSet events,
            preserving the chronological order of events and cast to
            ``List[Event]``.
        """
        slot_events = self._get_slot_set_events_for_changed_slots(
            agent_input, initial_slot_values, current_slot_values
        )
        return cast(
            List[Event],
            generated_events + slot_events,
        )

    def _generate_agent_task_completed_output(
        self,
        agent_input: AgentInput,
        slots: Dict[str, Any],
        tool_results: Dict[str, AgentToolResult],
        additional_events: Optional[List[Event]] = None,
    ) -> AgentOutput:
        """Generate an agent task completed output."""
        _slot_names_to_be_filled = self._get_slot_names_from_exit_conditions(
            agent_input
        )
        additional_slot_set_keys = {
            event.key
            for event in (additional_events or [])
            if isinstance(event, SlotSet) and event.key in _slot_names_to_be_filled
        }
        slot_events: List[Event] = [
            SlotSet(slot_name, slot_value)
            for slot_name, slot_value in slots.items()
            if (
                slot_name in _slot_names_to_be_filled
                and slot_name not in additional_slot_set_keys
            )
        ]
        # Combine additional events (like filler messages) with slot events
        all_events = (additional_events or []) + slot_events
        return AgentOutput(
            id=agent_input.id,
            status=AgentStatus.COMPLETED,
            events=all_events,
            structured_results=self._get_structured_results_for_agent_output(
                agent_input, tool_results
            ),
        )

    def _check_exit_conditions_after_iteration(
        self,
        agent_input: AgentInput,
        initial_slot_values: Dict[str, Any],
        current_slot_values: Dict[str, Any],
        tool_results: Dict[str, AgentToolResult],
        generated_events: Optional[List[Event]] = None,
        accumulated_tool_output_events: Optional[List[Event]] = None,
    ) -> Optional[AgentOutput]:
        """Evaluate task exit conditions using current slot values."""
        generated_events = generated_events or []
        accumulated_tool_output_events = accumulated_tool_output_events or []
        if not agent_input.metadata.get(AGENT_METADATA_EXIT_IF_KEY):
            return None

        structlogger.debug(
            "mcp_task_agent.exit_condition_check.start",
            agent_name=self._name,
            exit_conditions=agent_input.metadata[AGENT_METADATA_EXIT_IF_KEY],
        )
        exit_met, internal_error = self._is_exit_conditions_met(
            agent_input, current_slot_values
        )
        if internal_error:
            structlogger.error(
                "mcp_task_agent.exit_condition_check.error",
                agent_name=self._name,
                error=internal_error,
            )
            return AgentOutput(
                id=agent_input.id,
                status=AgentStatus.FATAL_ERROR,
                response_message=(
                    "An internal error occurred while checking the exit conditions."
                ),
                events=self.get_events_for_agent_output(
                    agent_input,
                    initial_slot_values,
                    current_slot_values,
                    generated_events + accumulated_tool_output_events,
                ),
                structured_results=self._get_structured_results_for_agent_output(
                    agent_input, tool_results
                ),
                error_message=internal_error,
            )

        structlogger.debug(
            "mcp_task_agent.exit_condition_check.result",
            agent_name=self._name,
            exit_met=exit_met,
        )
        if exit_met:
            return self._generate_agent_task_completed_output(
                agent_input,
                current_slot_values,
                tool_results,
                additional_events=generated_events + accumulated_tool_output_events,
            )

        return None

    def render_prompt_template(self, context: AgentInput) -> str:
        """Render the prompt template with the provided inputs."""
        # Build the context for the prompt.
        template_vars = self._build_context_for_prompt(context)
        template_vars["slot_names"] = self._get_slot_names_from_exit_conditions(context)
        # Render the prompt template.
        return Template(self.prompt_template).render(**template_vars)

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        """Send a message to the LLM and return the response.

        Each iteration generates and sends the LLM response via
        ``generate_and_send_response`` and routes:
        - no content and no tool calls → RECOVERABLE_ERROR
        - content only                 → content already streamed; return INPUT_REQUIRED
        - tool calls present           → execute tools; task-exit tools break the loop

        Any content streamed alongside tool calls is recorded as a BotUttered event
        in ``generated_events`` so the conversation history stays consistent.
        """
        message_build_cache: Dict[str, Any] = {}
        tool_call_messages: List[Dict[str, Any]] = []
        tool_results: Dict[str, AgentToolResult] = {}
        generated_events: List[Event] = []
        # Stores events returned by `process_tool_output`, accumulated across
        # all completed iterations in this `send_message` run.
        accumulated_tool_output_events: List[Event] = []

        _current_slot_values = {slot.name: slot.value for slot in agent_input.slots}
        _initial_slot_values = dict(_current_slot_values)
        _available_tools = self.get_available_tools(agent_input)
        _available_tools_names = [tool.name for tool in _available_tools]

        # Convert available tools to OpenAI JSON format
        tools_in_openai_format = [
            tool.to_litellm_json_format() for tool in _available_tools
        ]

        for iteration in range(self.MAX_ITERATIONS):
            current_iteration_tool_results: Dict[str, AgentToolResult] = {}
            try:
                messages = self._build_messages_for_llm_request_with_cache(
                    agent_input,
                    message_build_cache,
                )
                messages.extend(tool_call_messages)
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

                llm_response, bot_uttered = await self.generate_and_send_response(
                    messages=messages,
                    tools=tools_in_openai_format,
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
                        log_event_name="mcp_task_agent.send_message.no_llm_response",
                    )

                # Content only (no tool calls) → content already streamed;
                # return INPUT_REQUIRED
                if not llm_response.tool_calls:
                    # Exit conditions can already be satisfied without new tool calls
                    # (e.g. initial input).
                    if exit_output := self._check_exit_conditions_after_iteration(
                        agent_input=agent_input,
                        initial_slot_values=_initial_slot_values,
                        current_slot_values=_current_slot_values,
                        tool_results=tool_results,
                        generated_events=generated_events,
                        accumulated_tool_output_events=accumulated_tool_output_events,
                    ):
                        return exit_output
                    self._record_input_required_bot_uttered(
                        bot_uttered, generated_events
                    )
                    return self._create_input_required_output(
                        agent_input,
                        llm_content,
                        output_channel,
                        self.get_events_for_agent_output(
                            agent_input,
                            _initial_slot_values,
                            _current_slot_values,
                            generated_events + accumulated_tool_output_events,
                        ),
                        tool_results,
                    )

                if llm_response.tool_calls and bot_uttered:
                    self._record_filler_bot_uttered(bot_uttered, generated_events)

                # Add the assistant message with tool calls to the messages.
                tool_call_messages.append(
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

                    # If the tool is not available, return a fatal error output.
                    if tool_call.tool_name not in _available_tools_names:
                        return self._create_fatal_error_output(
                            agent_input,
                            f"Tool {tool_call.tool_name} is not available.",
                            "mcp_task_agent.send_message.tool_not_available",
                            events=self.get_events_for_agent_output(
                                agent_input,
                                _initial_slot_values,
                                _current_slot_values,
                                generated_events + accumulated_tool_output_events,
                            ),
                            tool_results=tool_results,
                            tool_name=tool_call.tool_name,
                        )

                    # If slot-setting tool, apply it and append the message to messages.
                    if slot_name := self._get_slot_name_from_tool_name(
                        tool_call.tool_name
                    ):
                        if error_output := self._handle_slot_setting_tool(
                            agent_input,
                            slot_name,
                            tool_call,
                            _current_slot_values,
                            _initial_slot_values,
                            tool_call_messages,
                            generated_events,
                            accumulated_tool_output_events,
                            tool_results,
                        ):
                            return error_output
                    else:
                        # Execute the tool call.
                        if error_output := await self._process_tool_call(
                            tool_call,
                            agent_input,
                            tool_call_messages,
                            tool_results,
                            current_iteration_tool_results,
                            "mcp_task_agent.send_message.tool_output",
                            events=self.get_events_for_agent_output(
                                agent_input,
                                _initial_slot_values,
                                _current_slot_values,
                                generated_events + accumulated_tool_output_events,
                            ),
                        ):
                            return error_output

                events_from_tool_results = await self._process_tool_output_or_raise(
                    current_iteration_tool_results,
                    tool_results,
                    output_channel,
                )
                if events_from_tool_results:
                    slot_updates = self._apply_slot_set_events_to_agent_input(
                        agent_input, events_from_tool_results
                    )
                    accumulated_tool_output_events.extend(events_from_tool_results)
                    agent_input.events.extend(events_from_tool_results)
                    _current_slot_values.update(slot_updates)

                if exit_output := self._check_exit_conditions_after_iteration(
                    agent_input=agent_input,
                    initial_slot_values=_initial_slot_values,
                    current_slot_values=_current_slot_values,
                    tool_results=tool_results,
                    generated_events=generated_events,
                    accumulated_tool_output_events=accumulated_tool_output_events,
                ):
                    return exit_output

            except Exception as e:
                if self._is_malformed_tool_response_exception(e):
                    # Continue to make another LLM call by breaking out of the current
                    # iteration and letting the loop continue with a fresh LLM request
                    self._append_malformed_tool_response_system_message(
                        tool_call_messages, agent_input, e, "mcp_task_agent"
                    )
                    continue
                return self._create_fatal_error_output(
                    agent_input,
                    str(e),
                    "mcp_task_agent.send_message.error_in_agent_loop",
                    events=self.get_events_for_agent_output(
                        agent_input,
                        _initial_slot_values,
                        _current_slot_values,
                        generated_events + accumulated_tool_output_events,
                    ),
                    tool_results=tool_results,
                )
        return self._create_max_iterations_reached_output(
            agent_input,
            events=self.get_events_for_agent_output(
                agent_input,
                _initial_slot_values,
                _current_slot_values,
                generated_events + accumulated_tool_output_events,
            ),
            tool_results=tool_results,
        )
