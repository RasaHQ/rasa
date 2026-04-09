from datetime import timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ListToolsResult, Tool
from mcp.types import Content, TextContent

from rasa.core.config.available_endpoints import (
    MCPFromSlotsEntry,
    MCPMetaMapConfig,
    MCPServerConfig,
)
from rasa.core.config.configuration import Configuration
from rasa.core.policies.flows.flow_step_result import ContinueFlowWithNextStep
from rasa.core.policies.flows.mcp_tool_executor import (
    _build_meta_for_flow_tool_call,
    _connect_to_mcp_server,
    _execute_mcp_tool_call,
    _get_meta_map_for_server,
    _get_slot_value_from_jinja2_expression,
    _handle_mcp_tool_error,
    _is_tool_available,
    _prepare_tool_arguments,
    _process_tool_result,
    call_mcp_tool,
)
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import McpToolExecuted, SlotSet
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps.call import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.mcp.server_connection import MCPServerConnection


@pytest.fixture
def mock_tracker() -> MagicMock:
    """Create a mock tracker for testing."""
    tracker = MagicMock(spec=DialogueStateTracker)
    tracker.get_slot.return_value = "test_value"
    return tracker


@pytest.fixture
def mock_stack() -> MagicMock:
    """Create a mock dialogue stack for testing."""
    stack = MagicMock(spec=DialogueStack)
    stack.push = MagicMock()
    return stack


@pytest.fixture
def mcp_call_step() -> CallFlowStep:
    """Create a CallFlowStep configured for MCP tool calling."""
    return CallFlowStep(
        custom_id="mcp_call_step",
        idx=0,
        description="MCP tool call step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_tool",
        mcp_server="test_server",
        mapping={
            "input": [{"slot": "test_slot", "param": "test_param"}],
            "output": [{"slot": "result_slot", "value": "result.content"}],
        },
    )


@pytest.fixture
def mock_mcp_server() -> MagicMock:
    """Create a mock MCP server for testing."""
    server = MagicMock()
    server.list_tools.return_value = ["test_tool", "other_tool"]
    server.call_tool = AsyncMock()
    return server


@pytest.fixture
def mock_mcp_connection() -> MagicMock:
    """Create a mock MCP connection for testing."""
    connection = MagicMock()
    connection.ensure_active_session = AsyncMock()
    return connection


@pytest.mark.parametrize(
    "tool_name,available_tools,expected_result",
    [
        (
            "test_tool",
            [
                Tool(name="test_tool", outputSchema={}, inputSchema={}),
                Tool(name="other_tool", outputSchema={}, inputSchema={}),
            ],
            True,
        ),
        (
            "test_tool",
            [Tool(name="other_tool", outputSchema={}, inputSchema={})],
            False,
        ),
        ("test_tool", [], False),
        ("", [Tool(name="test_tool", outputSchema={}, inputSchema={})], False),
    ],
)
@pytest.mark.asyncio
async def test_is_tool_available_success_cases(
    tool_name: str, available_tools: List[Tool], expected_result: bool
) -> None:
    """Test _is_tool_available with various tool availability scenarios."""
    mock_connection = MagicMock()
    mock_server = AsyncMock()
    mock_server.list_tools.return_value = ListToolsResult(tools=available_tools)
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_server)

    result = await _is_tool_available(mock_connection, tool_name)

    assert result == expected_result
    mock_connection.ensure_active_session.assert_called_once()
    mock_server.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_is_tool_available_exception_handling() -> None:
    """Test _is_tool_available when list_tools raises an exception."""
    mock_connection = MagicMock()
    mock_server = AsyncMock()
    # The exception should be raised when list_tools is called
    mock_server.list_tools.side_effect = Exception("Connection failed")
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_server)

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor.structlogger"
    ) as mock_logger:
        result = await _is_tool_available(mock_connection, "test_tool")

        assert result is False
        # Verify that ensure_active_session was called
        mock_connection.ensure_active_session.assert_called_once()
        # Verify that list_tools was called (and raised the exception)
        mock_server.list_tools.assert_called_once()
        # Verify that the warning was logged
        mock_logger.warning.assert_called_once()


@pytest.mark.parametrize(
    "input_mapping,slot_values,expected_arguments",
    [
        (
            [{"slot": "user_name", "param": "name"}],
            {"user_name": "John"},
            {"name": "John"},
        ),
        (
            [
                {"slot": "age", "param": "user_age"},
                {"slot": "city", "param": "location"},
            ],
            {"age": 25, "city": "New York"},
            {"user_age": 25, "location": "New York"},
        ),
        (
            [{"slot": "empty_slot", "param": "empty_param"}],
            {"empty_slot": None},
            {"empty_param": None},
        ),
        ([], {}, {}),
    ],
)
def test_prepare_tool_arguments(
    input_mapping: List[Dict[str, str]],
    slot_values: Dict[str, Any],
    expected_arguments: Dict[str, Any],
    mock_tracker: MagicMock,
) -> None:
    """Test _prepare_tool_arguments with various input mappings."""

    # Configure tracker to return different values for different slots
    def get_slot_side_effect(slot_name: str) -> Any:
        return slot_values.get(slot_name)

    mock_tracker.get_slot.side_effect = get_slot_side_effect

    result = _prepare_tool_arguments(input_mapping, mock_tracker)

    assert result == expected_arguments
    assert mock_tracker.get_slot.call_count == len(input_mapping)


@pytest.mark.parametrize(
    "result_content,output_mapping,expected_slot_sets",
    [
        (
            [TextContent(type="text", text="simple_string")],
            [{"slot": "result_slot", "value": "result.content"}],
            [
                SlotSet(
                    key="result_slot",
                    value=(
                        '[{"type": "text", "text": "simple_string", "annotations": null'
                        ', "meta": null}]'
                    ),
                )
            ],
        ),
        (
            [
                TextContent(type="text", text="simple_string"),
                TextContent(type="text", text="another_string"),
            ],
            [{"slot": "result_slot", "value": "result.content"}],
            [
                SlotSet(
                    key="result_slot",
                    value=(
                        '[{"type": "text", "text": "simple_string", "annotations": null'
                        ', "meta": null}, {"type": "text", "text": "another_string", '
                        '"annotations": null, "meta": null}]'
                    ),
                )
            ],
        ),
    ],
)
def test_process_tool_result_success_cases_with_content(
    result_content: List[Content],
    output_mapping: List[Dict[str, str]],
    expected_slot_sets: List[SlotSet],
) -> None:
    """Test _process_tool_result with various result contents."""
    # Create mock TextContent objects with model_dump method
    mock_content = []
    for content in result_content:
        mock_content_item = MagicMock()
        mock_content_item.model_dump.return_value = {
            "type": content.type,
            "text": content.text,
            "annotations": None,
            "meta": None,
        }
        mock_content.append(mock_content_item)

    # Create a mock CallToolResult with proper structure
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = mock_content
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": [item.model_dump() for item in mock_content],
        "structuredContent": None,
    }

    result = _process_tool_result(mock_result, output_mapping)

    assert isinstance(result, list)
    assert len(result) == len(expected_slot_sets)
    for i, slot_set in enumerate(result):
        assert isinstance(slot_set, SlotSet)
        assert slot_set.key == expected_slot_sets[i].key
        assert slot_set.value == expected_slot_sets[i].value


@pytest.mark.parametrize(
    "result_content,output_mapping,expected_slot_sets",
    [
        (
            {"key1": "value1", "key2": "value2"},
            [{"slot": "result_slot", "value": "result.structuredContent.key1"}],
            [SlotSet(key="result_slot", value="value1")],
        ),
        (
            {"key1": "value1", "key2": "value2"},
            [{"slot": "result_slot", "value": "result.structuredContent.key2"}],
            [SlotSet(key="result_slot", value="value2")],
        ),
    ],
)
def test_process_tool_result_success_cases_with_structured_content(
    result_content: Dict[str, Any],
    output_mapping: List[Dict[str, str]],
    expected_slot_sets: List[SlotSet],
) -> None:
    """Test _process_tool_result with various result contents."""
    # Create a mock CallToolResult with proper structure
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = None
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": None,
        "structuredContent": result_content,
    }

    result = _process_tool_result(mock_result, output_mapping)

    assert isinstance(result, list)
    assert len(result) == len(expected_slot_sets)
    for i, slot_set in enumerate(result):
        assert isinstance(slot_set, SlotSet)
        assert slot_set.key == expected_slot_sets[i].key
        assert slot_set.value == expected_slot_sets[i].value


def test_process_tool_result_empty_content() -> None:
    """Test _process_tool_result when content is empty."""
    # Create a mock CallToolResult with empty content
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = []
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": [],
        "structuredContent": None,
    }

    output_mapping = [{"slot": "output_slot", "value": "result.content"}]

    result = _process_tool_result(mock_result, output_mapping)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], SlotSet)
    assert result[0].key == "output_slot"
    assert result[0].value == []  # Empty list results in empty list


def test_handle_mcp_tool_error() -> None:
    """Test _handle_mcp_tool_error creates proper error handling."""
    mock_stack = MagicMock()
    events = [MagicMock()]
    error_message = "Test error message"
    tool_name = "test_tool"
    mcp_server = "test_server"

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor.structlogger"
    ) as mock_logger:
        result = _handle_mcp_tool_error(
            mock_stack, events, error_message, tool_name, mcp_server
        )

        # Verify error logging
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[1]
        assert call_args["error_message"] == error_message
        assert call_args["tool_name"] == tool_name
        assert call_args["mcp_server"] == mcp_server

        # Verify stack frame is pushed
        mock_stack.push.assert_called_once()
        pushed_frame = mock_stack.push.call_args[0][0]
        assert isinstance(pushed_frame, InternalErrorPatternFlowStackFrame)

        # Verify return type
        assert isinstance(result, ContinueFlowWithNextStep)
        assert result.events == events


@pytest.mark.asyncio
async def test_connect_to_mcp_server_success(
    mcp_call_step: CallFlowStep, mock_mcp_connection: MagicMock
) -> None:
    """Test _connect_to_mcp_server when connection is successful."""
    mock_endpoints = MagicMock()
    mock_endpoints.endpoints.mcp_servers = [
        MCPServerConfig(name="test_server", url="http://test:8080", type="http")
    ]

    with patch.object(Configuration, "get_instance", return_value=mock_endpoints):
        with patch.object(
            MCPServerConnection, "from_config", return_value=mock_mcp_connection
        ) as mock_from_config:
            # Mock ensure_active_session to return the connection itself
            mock_mcp_connection.ensure_active_session = AsyncMock()

            result = await _connect_to_mcp_server(mcp_call_step.mcp_server)

            assert result == mock_mcp_connection

            mock_mcp_connection.ensure_active_session.assert_called_once()

            mock_from_config.assert_called_once_with(
                {
                    "name": "test_server",
                    "url": "http://test:8080",
                    "type": "http",
                    "additional_params": {},
                    "meta_map": None,
                }
            )


@pytest.mark.asyncio
async def test_connect_to_mcp_server_success_with_auth(
    mcp_call_step: CallFlowStep, mock_mcp_connection: MagicMock
) -> None:
    """Test _connect_to_mcp_server when connection is successful."""
    mock_endpoints = MagicMock()
    mock_endpoints.endpoints.mcp_servers = [
        MCPServerConfig(
            name="test_server",
            url="http://test:8080",
            type="http",
            api_key="${TEST_TOKEN}",
        )
    ]

    with patch.object(Configuration, "get_instance", return_value=mock_endpoints):
        with patch.object(
            MCPServerConnection, "from_config", return_value=mock_mcp_connection
        ) as mock_from_config:
            # Mock ensure_active_session to return the connection itself
            mock_mcp_connection.ensure_active_session = AsyncMock()

            result = await _connect_to_mcp_server(mcp_call_step.mcp_server)

            assert result == mock_mcp_connection

            mock_mcp_connection.ensure_active_session.assert_called_once()

            mock_from_config.assert_called_once_with(
                {
                    "name": "test_server",
                    "url": "http://test:8080",
                    "type": "http",
                    "additional_params": {"api_key": "${TEST_TOKEN}"},
                    "meta_map": None,
                }
            )


@pytest.mark.asyncio
async def test_connect_to_mcp_server_no_servers_configured(
    mcp_call_step: CallFlowStep,
) -> None:
    """Test _connect_to_mcp_server when no MCP servers are configured."""
    mock_endpoints = MagicMock()
    mock_endpoints.endpoints.mcp_servers = []

    with patch.object(Configuration, "get_instance", return_value=mock_endpoints):
        result = await _connect_to_mcp_server(mcp_call_step)

        assert result is None


@pytest.mark.asyncio
async def test_connect_to_mcp_server_server_not_found(
    mcp_call_step: CallFlowStep,
) -> None:
    """Test _connect_to_mcp_server when the specified server is not found."""
    mock_endpoints = MagicMock()
    mock_endpoints.endpoints.mcp_servers = [
        MagicMock(name="other_server", url="http://other:8080", type="http")
    ]

    with patch.object(Configuration, "get_instance", return_value=mock_endpoints):
        result = await _connect_to_mcp_server(mcp_call_step)

        assert result is None


@pytest.mark.asyncio
async def test_connect_to_mcp_server_multiple_matching_servers(
    mcp_call_step: CallFlowStep,
) -> None:
    """Test _connect_to_mcp_server when multiple servers match the name."""
    mock_endpoints = MagicMock()
    mock_endpoints.endpoints.mcp_servers = [
        MagicMock(name="test_server", url="http://test1:8080", type="http"),
        MagicMock(name="test_server", url="http://test2:8080", type="http"),
    ]

    with patch.object(Configuration, "get_instance", return_value=mock_endpoints):
        result = await _connect_to_mcp_server(mcp_call_step)

        assert result is None


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_success(
    mcp_call_step: CallFlowStep,
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
    mock_mcp_server: MagicMock,
) -> None:
    """Test _execute_mcp_tool_call with successful tool execution."""
    initial_events = []

    # Create a proper mock CallToolResult
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = [{"result": "success"}]
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": [{"result": "success"}],
        "structuredContent": None,
    }

    # Create a mock connection that returns a mock server
    mock_connection = MagicMock()
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_mcp_server)
    mock_connection.close = AsyncMock()  # Mock close as AsyncMock since it's awaited
    mock_mcp_server.call_tool.return_value = mock_result

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._connect_to_mcp_server",
        return_value=mock_connection,
    ):
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._is_tool_available",
            return_value=True,
        ):
            with patch(
                "rasa.core.policies.flows.mcp_tool_executor._process_tool_result"
            ) as mock_process:
                mock_process.return_value = [
                    SlotSet(key="result_slot", value='{"result": "success"}')
                ]

                result = await _execute_mcp_tool_call(
                    initial_events, mock_stack, mcp_call_step, mock_tracker
                )

                assert isinstance(result, ContinueFlowWithNextStep)
                assert len(result.events) == 2
                assert isinstance(result.events[0], McpToolExecuted)
                assert isinstance(result.events[1], SlotSet)
                mock_connection.ensure_active_session.assert_called_once()
                mock_mcp_server.call_tool.assert_called_once_with(
                    "test_tool",
                    {"test_param": "test_value"},
                    read_timeout_seconds=timedelta(seconds=10),
                )
                # Verify connection cleanup
                mock_connection.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_empty_result(
    mcp_call_step: CallFlowStep,
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
    mock_mcp_server: MagicMock,
) -> None:
    """Test _execute_mcp_tool_call when tool returns empty result."""
    initial_events = []
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = None
    mock_result.structuredContent = None

    # Create a mock connection that returns a mock server
    mock_connection = MagicMock()
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_mcp_server)
    mock_connection.close = AsyncMock()  # Mock close as AsyncMock since it's awaited
    mock_mcp_server.call_tool.return_value = mock_result

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._connect_to_mcp_server",
        return_value=mock_connection,
    ):
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._is_tool_available",
            return_value=True,
        ):
            with patch(
                "rasa.core.policies.flows.mcp_tool_executor.structlogger"
            ) as mock_logger:
                result = await _execute_mcp_tool_call(
                    initial_events, mock_stack, mcp_call_step, mock_tracker
                )

                assert isinstance(result, ContinueFlowWithNextStep)
                assert result.events == initial_events
                mock_logger.warning.assert_called_once()
                # Verify connection cleanup
                mock_connection.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_mcp_tool_call_structured_content_only(
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
    mock_mcp_server: MagicMock,
) -> None:
    """Test _execute_mcp_tool_call when tool returns only structured content."""
    initial_events = []
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = None
    mock_result.structuredContent = {"answer": "from_structured"}
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": None,
        "structuredContent": {"answer": "from_structured"},
    }

    step = CallFlowStep(
        custom_id="test_step",
        idx=0,
        description="Step with structuredContent output",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_tool",
        mcp_server="test_server",
        mapping={
            "input": [],
            "output": [
                {"slot": "result_slot", "value": "result.structuredContent.answer"}
            ],
        },
    )

    mock_connection = MagicMock()
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_mcp_server)
    mock_connection.close = AsyncMock()
    mock_mcp_server.call_tool.return_value = mock_result

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._connect_to_mcp_server",
        return_value=mock_connection,
    ):
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._is_tool_available",
            return_value=True,
        ):
            result = await _execute_mcp_tool_call(
                initial_events, mock_stack, step, mock_tracker
            )

    assert isinstance(result, ContinueFlowWithNextStep)
    assert len(result.events) == 2
    assert isinstance(result.events[0], McpToolExecuted)
    assert isinstance(result.events[1], SlotSet)
    assert result.events[1].key == "result_slot"
    assert result.events[1].value == "from_structured"
    mock_connection.close.assert_called_once()


@pytest.mark.asyncio
async def test_call_mcp_tool_success(
    mcp_call_step: CallFlowStep,
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
) -> None:
    """Test call_mcp_tool with successful execution."""
    initial_events = []

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call"
    ) as mock_execute:
        mock_execute.return_value = ContinueFlowWithNextStep(events=initial_events)

        with patch(
            "rasa.core.policies.flows.mcp_tool_executor.structlogger"
        ) as mock_logger:
            result = await call_mcp_tool(
                initial_events, mock_stack, mcp_call_step, mock_tracker
            )

            assert isinstance(result, ContinueFlowWithNextStep)
            mock_execute.assert_called_once_with(
                initial_events,
                mock_stack,
                mcp_call_step,
                mock_tracker,
            )
            mock_logger.debug.assert_called_once()


@pytest.mark.asyncio
async def test_call_mcp_tool_exception_handling(
    mcp_call_step: CallFlowStep,
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
) -> None:
    """Test call_mcp_tool when _execute_mcp_tool_call raises an exception."""
    initial_events = []
    test_exception = Exception("Test exception")

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call",
        side_effect=test_exception,
    ):
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._handle_mcp_tool_error"
        ) as mock_error_handler:
            mock_error_handler.return_value = ContinueFlowWithNextStep(
                events=initial_events
            )

            result = await call_mcp_tool(
                initial_events, mock_stack, mcp_call_step, mock_tracker
            )

            mock_error_handler.assert_called_once()
            # The function is called with positional args for first 3 params,
            # then keyword args
            keyword_args = mock_error_handler.call_args[1]
            assert (
                "Failed to execute MCP tool call: Test exception."
                in keyword_args["error_message"]
            )
            assert keyword_args["tool_name"] == "test_tool"
            assert keyword_args["mcp_server"] == "test_server"
            assert isinstance(result, ContinueFlowWithNextStep)


@pytest.mark.parametrize(
    "mapping_structure,expected_input_mapping",
    [
        (
            {
                "input": [{"slot": "slot1", "param": "param1"}],
                "output": [{"slot": "output_slot", "value": "result.content"}],
            },
            [{"slot": "slot1", "param": "param1"}],
        ),
        (
            {
                "input": [],
                "output": [{"slot": "output_slot", "value": "result.content"}],
            },
            [],
        ),
        (
            {
                "input": [{"slot": "s1", "param": "p1"}, {"slot": "s2", "param": "p2"}],
                "output": [{"slot": "output_slot", "value": "result.content"}],
            },
            [{"slot": "s1", "param": "p1"}, {"slot": "s2", "param": "p2"}],
        ),
    ],
)
def test_execute_mcp_tool_call_with_different_mappings(
    mapping_structure: Dict[str, Any],
    expected_input_mapping: List[Dict[str, str]],
    mock_tracker: MagicMock,
    mock_stack: MagicMock,
    mock_mcp_server: MagicMock,
) -> None:
    """Test _execute_mcp_tool_call with different mapping structures."""
    step = CallFlowStep(
        custom_id="test_step",
        idx=0,
        description="Test step",
        metadata={},
        next=FlowStepLinks(links=[]),
        flow_id="test_flow",
        call="test_tool",
        mcp_server="test_server",
        mapping=mapping_structure,
    )

    initial_events = []

    # Create a proper mock CallToolResult
    mock_result = MagicMock()
    mock_result.isError = False
    mock_result.content = [{"result": "success"}]
    mock_result.model_dump.return_value = {
        "isError": False,
        "content": [{"result": "success"}],
        "structuredContent": None,
    }

    # Create a mock connection that returns a mock server
    mock_connection = MagicMock()
    mock_connection.ensure_active_session = AsyncMock(return_value=mock_mcp_server)
    mock_connection.close = AsyncMock()  # Mock close as AsyncMock since it's awaited
    mock_mcp_server.call_tool.return_value = mock_result

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._connect_to_mcp_server",
        return_value=mock_connection,
    ):
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._is_tool_available",
            return_value=True,
        ):
            with patch(
                "rasa.core.policies.flows.mcp_tool_executor._process_tool_result"
            ) as mock_process:
                mock_process.return_value = [
                    SlotSet(key="output_slot", value='{"result": "success"}')
                ]

                # This is an async function, so we need to run it in an event loop
                import asyncio

                async def run_test():
                    return await _execute_mcp_tool_call(
                        initial_events, mock_stack, step, mock_tracker
                    )

                result = asyncio.run(run_test())

                assert isinstance(result, ContinueFlowWithNextStep)
                # Verify that the tool was called with the correct arguments based
                # on the mapping
                if expected_input_mapping:
                    expected_args = {
                        mapping["param"]: "test_value"
                        for mapping in expected_input_mapping
                    }
                    mock_connection.ensure_active_session.assert_called_once()
                    mock_mcp_server.call_tool.assert_called_once_with(
                        "test_tool",
                        expected_args,
                        read_timeout_seconds=timedelta(seconds=10),
                    )
                else:
                    mock_connection.ensure_active_session.assert_called_once()
                    mock_mcp_server.call_tool.assert_called_once_with(
                        "test_tool", {}, read_timeout_seconds=timedelta(seconds=10)
                    )

                # Verify connection cleanup
                mock_connection.close.assert_called_once()


@pytest.mark.parametrize(
    "expression,expected,context_name",
    [
        # Simple dot notation access
        ("result.content", "Hello World", "basic"),
        ("result.structuredContent.user.name", "John", "basic"),
        ("result.structuredContent.user.age", 30, "basic"),
        ("result.structuredContent.status", "active", "basic"),
        # List/array access with indices
        ("result.structuredContent[0].text", "First item", "list_access"),
        ("result.structuredContent[1].id", 2, "list_access"),
        ("result.content.fruits[2]", "cherry", "list_access"),
        # String methods
        ("result.content.upper()", "HELLO WORLD", "string_methods"),
        ("result.content.title()", "Hello World", "string_methods"),
        ("result.user.name.title()", "John Doe", "string_methods"),
        # Conditional expressions
        ("result.content if result.content else 'No content'", "Hello", "conditional"),
        (
            "result.empty_field if result.empty_field else 'Default'",
            "Default",
            "conditional",
        ),
        ("result.user.name if result.user else 'Anonymous'", "John", "conditional"),
        # Jinja2 filters
        ("result.content.title | upper", "HELLO WORLD", "filters"),
        ("result.content.fruits | length", 3, "filters"),
        ("result.content.numbers | sum", 15, "filters"),
        # Default filter
        ("result.content | default('No content')", "Hello", "default_filter"),
        ("result.missing_field | default('somevalue')", None, "default_filter"),
        ("result.nonexistent | default('Fallback')", "Fallback", "default_filter"),
        # Mathematical operations
        ("result.numbers.a + result.numbers.b", 15, "math"),
        ("result.numbers.a - result.numbers.b", 5, "math"),
        ("result.numbers.a * result.numbers.c", 20, "math"),
        ("result.numbers.a / result.numbers.b", 2.0, "math"),
        # Complex nested structures
        ("result.users[0].name", "John", "complex"),
        ("result.users[1].active", False, "complex"),
        ("result.metadata.total", 3, "complex"),
        ("result.metadata.active_count", 2, "complex"),
        # Boolean operations
        ("result.user.age > result.threshold", True, "advanced"),
        ("result.user.active and result.user.age > 20", True, "advanced"),
        ("result.user.age < 20 or result.user.active", True, "advanced"),
        # List operations
        ("result.fruits[0]", "apple", "advanced"),
        ("result.fruits[-1]", "cherry", "advanced"),
        ("result.numbers | max", 5, "advanced"),
        ("result.numbers | min", 1, "advanced"),
        # None values
        ("result.content", None, "advanced"),
        ("result.user.name", None, "advanced"),
        ("result.user.age", 25, "advanced"),
        # Special characters in keys
        ("result.items[1]", None, "special_chars"),  # Does not work with dot notation
        ("result['items'][1]", "banana", "special_chars"),  # Works with square brackets
    ],
)
def test_get_slot_value_comprehensive(
    expression: str, expected: Any, context_name: str
) -> None:
    """Comprehensive test for all Jinja2 expression functionality."""
    # Define all contexts
    contexts = {
        "basic": {
            "result": {
                "content": "Hello World",
                "structuredContent": {
                    "user": {"name": "John", "age": 30},
                    "status": "active",
                },
            }
        },
        "list_access": {
            "result": {
                "structuredContent": [
                    {"text": "First item", "id": 1},
                    {"text": "Second item", "id": 2},
                    {"text": "Third item", "id": 3},
                ],
                "content": {"fruits": ["apple", "banana", "cherry"]},
            }
        },
        "string_methods": {
            "result": {"content": "hello world", "user": {"name": "john doe"}}
        },
        "conditional": {
            "result": {
                "content": "Hello",
                "user": {"name": "John"},
                "empty_field": None,
            }
        },
        "filters": {
            "result": {
                "content": {
                    "title": "hello world",
                    "fruits": ["apple", "banana", "cherry"],
                    "numbers": [1, 2, 3, 4, 5],
                }
            }
        },
        "default_filter": {"result": {"content": "Hello", "missing_field": None}},
        "math": {
            "result": {"numbers": {"a": 10, "b": 5, "c": 2}, "list": [1, 2, 3, 4, 5]}
        },
        "complex": {
            "result": {
                "users": [
                    {"name": "John", "age": 30, "active": True},
                    {"name": "Jane", "age": 25, "active": False},
                    {"name": "Bob", "age": 35, "active": True},
                ],
                "metadata": {"total": 3, "active_count": 2},
            }
        },
        "advanced": {
            "result": {
                "user": {"age": 25, "active": True, "name": None},
                "threshold": 18,
                "fruits": ["apple", "banana", "cherry"],
                "numbers": [1, 2, 3, 4, 5],
                "content": None,
            }
        },
        "special_chars": {"result": {"items": ["apple", "banana", "cherry"]}},
    }

    context = contexts[context_name]
    assert _get_slot_value_from_jinja2_expression(expression, context) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "result.nonexistent.field",  # Should raise jinja2.exceptions.UndefinedError
        "result.content[",  # Should raise jinja2.exceptions.TemplateError
    ],
)
def test_get_slot_value_error_cases(expression: str) -> None:
    """Test that invalid expressions raise appropriate errors."""
    context = {"result": {"content": "Hello"}}

    with pytest.raises(Exception):
        _get_slot_value_from_jinja2_expression(expression, context)


def test_get_slot_value_empty_context() -> None:
    """Test behavior with empty context."""
    context = {}

    # Test with empty context
    with pytest.raises(Exception):
        _get_slot_value_from_jinja2_expression("result.content", context)


def test_build_meta_for_flow_tool_call_with_meta_map(
    mock_tracker: MagicMock,
) -> None:
    """Test that _build_meta_for_flow_tool_call returns slot values keyed by param."""
    mock_tracker.slots = {"user_id": MagicMock(), "role": MagicMock()}
    mock_tracker.current_slot_values.return_value = {
        "user_id": "user_123",
        "role": "admin",
    }
    mock_endpoints = MagicMock()
    mock_endpoints.mcp_servers = [
        MCPServerConfig(
            name="test_server",
            url="http://test:8080",
            type="http",
            meta_map=MCPMetaMapConfig(
                from_slots=[
                    MCPFromSlotsEntry(slot="user_id", param="user_id"),
                    MCPFromSlotsEntry(slot="role", param="user_role"),
                ],
            ),
        )
    ]
    with patch.object(
        Configuration, "get_instance", return_value=MagicMock(endpoints=mock_endpoints)
    ):
        meta = _build_meta_for_flow_tool_call("test_server", mock_tracker)
    assert meta == {
        "user_id": "user_123",
        "user_role": "admin",
    }


@pytest.mark.asyncio
async def test_get_meta_map_for_server_returns_none_when_no_servers() -> None:
    """Test _get_meta_map_for_server returns None when no MCP servers configured."""
    mock_endpoints = MagicMock()
    mock_endpoints.mcp_servers = None
    with patch.object(
        Configuration, "get_instance", return_value=MagicMock(endpoints=mock_endpoints)
    ):
        assert _get_meta_map_for_server("any_server") is None
