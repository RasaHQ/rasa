import json
from datetime import timedelta
from typing import Any, Dict, List, Optional

import structlog
from jinja2.sandbox import SandboxedEnvironment
from mcp.types import CallToolResult

from rasa.core.config.configuration import Configuration
from rasa.core.policies.flows.flow_step_result import (
    ContinueFlowWithNextStep,
    FlowStepResult,
)
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.mcp.server_connection import MCPServerConnection
from rasa.utils.common import ensure_jsonified_iterable

structlogger = structlog.get_logger()

CONFIG_VALUE = "value"
CONFIG_SLOT = "slot"
TOOL_CALL_DEFATULT_TIMEOUT = 10  # seconds


async def call_mcp_tool(
    initial_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    """Run an MCP tool call step."""
    structlogger.debug(
        "flow.step.call_mcp_tool",
        tool_id=step.call,
        mcp_server=step.mcp_server,
        mapping=step.mapping,
        step_id=step.id,
        flow_id=step.flow_id,
        json_formatting=["mapping"],
    )

    try:
        return await _execute_mcp_tool_call(initial_events, stack, step, tracker)
    except Exception as e:
        return _handle_mcp_tool_error(
            stack,
            initial_events,
            error_message=f"Failed to execute MCP tool call: {e}.",
            tool_name=step.call,
            mcp_server=step.mcp_server,
        )


async def _execute_mcp_tool_call(
    initial_events: List[Event],
    stack: DialogueStack,
    step: CallFlowStep,
    tracker: DialogueStateTracker,
) -> FlowStepResult:
    """Execute the MCP tool call with proper error handling."""
    mcp_server_connection = None
    try:
        # Connect to the MCP server
        mcp_server_connection = await _connect_to_mcp_server(step.mcp_server)

        if not mcp_server_connection:
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"Cannot connect to MCP server '{step.mcp_server}'.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
            )

        # Validate tool availability
        if not await _is_tool_available(mcp_server_connection, step.call):
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"Tool '{step.call}' is not available on MCP server "
                f"'{step.mcp_server}'.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
            )

        # This should not happen, but we need to check for type checking to pass
        if not step.mapping:
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"No mapping found for tool '{step.call}'.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
            )

        # Prepare arguments for the tool call
        arguments = _prepare_tool_arguments(step.mapping["input"], tracker)

        # Call the tool with parameters
        mcp_server = await mcp_server_connection.ensure_active_session()
        result = await mcp_server.call_tool(
            step.call,
            arguments,
            read_timeout_seconds=timedelta(seconds=TOOL_CALL_DEFATULT_TIMEOUT),
        )

        # Handle tool execution result
        if result is None or result.isError:
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"Tool '{step.call}' execution failed: {result.content}.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
            )
        elif not result.content:
            structlogger.warning(
                "call_mcp_tool.empty_tool_result",
                tool_name=step.call,
                mcp_server=step.mcp_server,
            )
        else:
            structlogger.debug(
                "call_mcp_tool.tool_execution_success",
                tool_name=step.call,
                mcp_server=step.mcp_server,
                result_content=result.content,
                result_structured_content=result.structuredContent,
                json_formatting=["result_content", "result_structured_content"],
            )

            # Process successful result
            if set_slot_event := _process_tool_result(result, step.mapping["output"]):
                initial_events.extend(set_slot_event)
            else:
                return _handle_mcp_tool_error(
                    stack,
                    initial_events,
                    f"Failed to process tool result for '{step.call}'.",
                    tool_name=step.call,
                    mcp_server=step.mcp_server,
                )

        return ContinueFlowWithNextStep(events=initial_events)

    finally:
        # Always clean up the connection to prevent resource leaks
        if mcp_server_connection:
            try:
                await mcp_server_connection.close()
            except Exception as e:
                structlogger.warning(
                    "call_mcp_tool.connection_cleanup_failed",
                    tool_name=step.call,
                    mcp_server=step.mcp_server,
                    error=str(e),
                )


async def _is_tool_available(
    mcp_server_connection: MCPServerConnection, tool_name: str
) -> bool:
    """Check if the specified tool is available on the MCP server."""
    try:
        # Get the active session from the connection
        mcp_server = await mcp_server_connection.ensure_active_session()
        available_tools = await mcp_server.list_tools()
        tool_names = [tool.name for tool in available_tools.tools]
        return tool_name in tool_names
    except Exception as e:
        structlogger.warning(
            "call_mcp_tool.tool_availability_check_failed",
            tool_name=tool_name,
            error=str(e),
        )
        return False


async def _connect_to_mcp_server(
    mcp_server_name: Optional[str],
) -> Optional[MCPServerConnection]:
    """Connect to the MCP server."""
    if not mcp_server_name:
        return None

    # get the MCP server config from the available endpoints
    endpoints = Configuration.get_instance().endpoints
    mcp_servers = endpoints.mcp_servers
    if not mcp_servers:
        return None

    mcp_server_configs = [
        mcp_server for mcp_server in mcp_servers if mcp_server.name == mcp_server_name
    ]
    if not mcp_server_configs or len(mcp_server_configs) != 1:
        return None

    mcp_server_config = mcp_server_configs[0]

    mcp_server_connection = MCPServerConnection.from_config(
        mcp_server_config.model_dump()
    )

    # Ensure the connection is established and return the connection object
    await mcp_server_connection.ensure_active_session()
    return mcp_server_connection


def _prepare_tool_arguments(
    input_mapping: List[Dict[str, str]], tracker: DialogueStateTracker
) -> Dict[str, Any]:
    """Prepare arguments for the tool call from slot values."""
    arguments = {}
    for argument in input_mapping:
        slot_value = tracker.get_slot(argument["slot"])
        arguments[argument["param"]] = slot_value
    return arguments


def _jsonify_slot_value(value: Any) -> str | int | float | bool | None:
    """Prepare value for SlotSet: iterables -> JSON string, primitives -> as-is"""
    if isinstance(value, (list, dict)) and len(value):
        return json.dumps(ensure_jsonified_iterable(value))
    return value


def _get_slot_value_from_jinja2_expression(
    result_expression: str,
    result_as_dict: Dict[str, Any],
) -> Any:
    """Get the slot value from the Jinja2 expression"""
    # Create a sandboxed environment to evaluate the expression
    _env = SandboxedEnvironment()

    # Compile the expression
    compiled_expr = _env.compile_expression(result_expression)

    # Evaluate the expression
    return compiled_expr(result_as_dict)


def _process_tool_result(
    result: CallToolResult,
    output_mapping: List[Dict[str, str]],
) -> Optional[List[SlotSet]]:
    """Create a SetSlot event for the tool result using Jinja2 expressions"""
    try:
        _result_as_dict = {"result": result.model_dump()}
        slots = []
        for mapping in output_mapping:
            try:
                result_expression = mapping[CONFIG_VALUE]

                # Get the slot value from the Jinja2 expression
                slot_value = _get_slot_value_from_jinja2_expression(
                    result_expression, _result_as_dict
                )

                slots.append(
                    SlotSet(mapping[CONFIG_SLOT], _jsonify_slot_value(slot_value))
                )
            except Exception as e:
                structlogger.error(
                    "call_mcp_tool.value_not_found_in_tool_result",
                    slot=mapping[CONFIG_SLOT],
                    value=mapping[CONFIG_VALUE],
                    result=_result_as_dict,
                    error=str(e),
                    json_formatting=["result"],
                )
                return None
        return slots
    except Exception as e:
        structlogger.error(
            "call_mcp_tool.result_processing_failed",
            error=str(e),
            result=result,
            json_formatting=["result"],
        )
        return None


def _handle_mcp_tool_error(
    stack: DialogueStack,
    events: List[Event],
    error_message: str,
    tool_name: str,
    mcp_server: Optional[str],
) -> FlowStepResult:
    """Handle MCP tool errors consistently."""
    structlogger.error(
        "call_mcp_tool.error",
        error_message=error_message,
        tool_name=tool_name,
        mcp_server=mcp_server,
    )
    stack.push(InternalErrorPatternFlowStackFrame())
    return ContinueFlowWithNextStep(events=events)
