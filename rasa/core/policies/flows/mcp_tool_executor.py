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
    INTERNAL_ERROR_SOURCE_MCP_TOOL,
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import Event, McpToolExecuted, SlotSet
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.mcp.server_connection import MCPServerConnection
from rasa.shared.utils.mcp.utils import build_mcp_meta, call_tool_with_meta
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
    """Run an MCP tool call step.

    Args:
        initial_events: Events collected so far.
        stack: Dialogue stack.
        step: The call flow step.
        tracker: Dialogue state tracker.
    """
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
            flow_id=step.flow_id,
            step_id=step.id,
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
                flow_id=step.flow_id,
                step_id=step.id,
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
                flow_id=step.flow_id,
                step_id=step.id,
            )

        # This should not happen, but we need to check for type checking to pass
        if not step.mapping:
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"No mapping found for tool '{step.call}'.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
                flow_id=step.flow_id,
                step_id=step.id,
            )

        # Prepare arguments for the tool call
        arguments = _prepare_tool_arguments(step.mapping["input"], tracker)

        # Build _meta from meta_map config (not visible to the LLM)
        meta = _build_meta_for_flow_tool_call(step.mcp_server, tracker)

        # Call the tool with parameters
        mcp_server = await mcp_server_connection.ensure_active_session()
        result: CallToolResult = await call_tool_with_meta(
            mcp_server,
            step.call,
            arguments,
            timedelta(seconds=TOOL_CALL_DEFATULT_TIMEOUT),
            meta,
        )

        initial_events.append(_create_mcp_tool_executed_event(step, arguments, result))

        # Handle tool execution result
        if result is None or result.isError:
            return _handle_mcp_tool_error(
                stack,
                initial_events,
                f"Tool '{step.call}' execution failed: {result.content}.",
                tool_name=step.call,
                mcp_server=step.mcp_server,
                flow_id=step.flow_id,
                step_id=step.id,
            )
        elif not result.content and not result.structuredContent:
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

            # Process successful result (from content and/or structuredContent)
            if set_slot_event := _process_tool_result(result, step.mapping["output"]):
                initial_events.extend(set_slot_event)
            else:
                return _handle_mcp_tool_error(
                    stack,
                    initial_events,
                    f"Failed to process tool result for '{step.call}'.",
                    tool_name=step.call,
                    mcp_server=step.mcp_server,
                    flow_id=step.flow_id,
                    step_id=step.id,
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


def _get_meta_map_for_server(mcp_server_name: Optional[str]) -> Optional[Any]:
    """Get meta_map config for an MCP server from endpoints."""
    if not mcp_server_name:
        return None
    endpoints = Configuration.get_instance().endpoints
    if not endpoints.mcp_servers:
        return None
    for server in endpoints.mcp_servers:
        if server.name == mcp_server_name:
            return server.meta_map
    return None


def _build_meta_for_flow_tool_call(
    mcp_server_name: Optional[str],
    tracker: DialogueStateTracker,
) -> Dict[str, Any]:
    """Build _meta dict for a flow-based MCP tool call from meta_map and tracker slots.

    Slot presence in the domain is validated at training time; no runtime check needed.
    """
    meta_map = _get_meta_map_for_server(mcp_server_name)
    if not meta_map:
        return {}
    slots = tracker.current_slot_values()
    return build_mcp_meta(meta_map, slots)


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
    """Prepare value for SlotSet: iterables -> JSON string, primitives -> as-is."""
    if isinstance(value, (list, dict)) and len(value):
        return json.dumps(ensure_jsonified_iterable(value))
    return value


def _get_slot_value_from_jinja2_expression(
    result_expression: str,
    result_as_dict: Dict[str, Any],
) -> Any:
    """Get the slot value from the Jinja2 expression."""
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
    """Create a SetSlot event for the tool result using Jinja2 expressions."""
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


def _serialize_tool_result_for_event(result: Optional[CallToolResult]) -> Any:
    """Serialize a tool result into a JSON-serializable event payload."""
    if result is None:
        return None

    try:
        return result.model_dump()
    except Exception:
        return {"content": str(result.content), "structuredContent": None}


def _create_mcp_tool_executed_event(
    step: CallFlowStep,
    arguments: Dict[str, Any],
    result: Optional[CallToolResult],
) -> McpToolExecuted:
    """Create an inspector event for a flow MCP tool execution."""
    metadata = {
        key: value
        for key, value in {
            "flow_id": step.flow_id,
            "step_id": step.id,
        }.items()
        if value is not None
    }

    return McpToolExecuted(
        tool_name=step.call,
        arguments=arguments,
        result=(
            None
            if result is None or result.isError
            else _serialize_tool_result_for_event(result)
        ),
        is_error=result is None or result.isError,
        error_message=(
            str(result.content) if result is not None and result.isError else None
        ),
        metadata=metadata or None,
    )


def _handle_mcp_tool_error(
    stack: DialogueStack,
    events: List[Event],
    error_message: str,
    tool_name: str,
    mcp_server: Optional[str],
    flow_id: Optional[str] = None,
    step_id: Optional[str] = None,
) -> FlowStepResult:
    """Handle MCP tool errors consistently.

    Emits a ``McpToolExecuted`` error event (when one for the same tool has not
    already been appended) so that the inspector frontend can display the
    specific failure.
    """
    structlogger.error(
        "call_mcp_tool.error",
        error_message=error_message,
        tool_name=tool_name,
        mcp_server=mcp_server,
    )
    if not any(
        isinstance(e, McpToolExecuted) and e.tool_name == tool_name for e in events
    ):
        metadata = {
            k: v
            for k, v in {
                "flow_id": flow_id,
                "step_id": step_id,
                "mcp_server": mcp_server,
            }.items()
            if v is not None
        }
        events.append(
            McpToolExecuted(
                tool_name=tool_name,
                arguments={},
                result=None,
                is_error=True,
                error_message=error_message,
                metadata=metadata or None,
            )
        )
    error_info: Dict[str, Any] = {
        "error_source": INTERNAL_ERROR_SOURCE_MCP_TOOL,
        "tool_name": tool_name,
        "mcp_server": mcp_server,
        "error_message": error_message,
    }
    if flow_id is not None:
        error_info["flow_id"] = flow_id
    if step_id is not None:
        error_info["step_id"] = step_id
    stack.push(InternalErrorPatternFlowStackFrame(info=error_info))
    return ContinueFlowWithNextStep(events=events)
