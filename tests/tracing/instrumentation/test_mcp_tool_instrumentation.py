import importlib
import json
from typing import Any, List, Sequence
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.core.policies.flows.flow_step_result import ContinueFlowWithNextStep
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import SessionStarted, SlotSet
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events("test", evts=[SessionStarted()])


@pytest.fixture
def dialogue_stack() -> DialogueStack:
    return DialogueStack(frames=[])


@pytest.fixture
def mcp_call_flow_step() -> CallFlowStep:
    """Create a CallFlowStep specifically for MCP tool testing."""
    return CallFlowStep(
        idx=1,
        custom_id="test_mcp_call_step",
        description="Test MCP tool step",
        call="test_tool",
        mcp_server="test_server",
        flow_id="test_flow",
        mapping={
            "input": [{"slot": "test_slot", "param": "test_param"}],
            "output": [{"slot": "result_slot", "param": "result"}],
        },
        metadata={},
        next=FlowStepLinks(links=[]),
    )


@pytest.mark.asyncio
async def test_tracing_mcp_tool_execution(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    mcp_call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
) -> None:
    """Test tracing for MCP tool execution in flow steps."""
    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._connect_to_mcp_server"
    ) as mock_connect:
        with patch(
            "rasa.core.policies.flows.mcp_tool_executor._is_tool_available"
        ) as mock_available:
            with patch(
                "rasa.shared.utils.mcp.server_connection.MCPServerConnection.ensure_active_session"
            ) as mock_session:
                instrumentation.instrument(tracer_provider)

                mock_connection = MagicMock()
                mock_connect.return_value = mock_connection
                mock_available.return_value = True

                mock_mcp_server = MagicMock()
                mock_session.return_value = mock_mcp_server

                mock_tool_result = MagicMock()
                mock_tool_result.content = [{"text": "test result"}]
                mock_tool_result.structuredContent = None
                mock_tool_result.model_dump.return_value = {
                    "content": [{"text": "test result"}]
                }
                mock_mcp_server.call_tool.return_value = mock_tool_result

                module = importlib.import_module(
                    "rasa.core.policies.flows.mcp_tool_executor"
                )
                call_mcp_tool = getattr(module, "call_mcp_tool")

                initial_events: List[Any] = []
                await call_mcp_tool(
                    initial_events, dialogue_stack, mcp_call_flow_step, tracker
                )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans > 0

    tool_spans = [
        span
        for span in captured_spans
        if span.name
        == "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call"
    ]
    assert len(tool_spans) > 0

    tool_span = tool_spans[0]
    assert tool_span.attributes is not None
    assert "tool_id" in tool_span.attributes
    assert "tool_input_arguments" in tool_span.attributes
    assert tool_span.attributes["execution_context"] == "flow"


@pytest.mark.asyncio
async def test_histogram_recording_mcp_tool_execution(
    tracer_provider: TracerProvider,
    tracker: DialogueStateTracker,
    mcp_call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
) -> None:
    """Test that histogram recording actually works for MCP tool execution."""
    # Mock the histogram to verify record() is called
    mock_histogram = Mock()

    # Mock the MetricInstrumentProvider to return our mock histogram
    with patch.object(
        MetricInstrumentProvider, "get_instrument", return_value=mock_histogram
    ):
        with patch.object(
            MetricInstrumentProvider, "instruments", {"test": "instruments"}
        ):
            # Mock the instrumented function BEFORE instrumentation
            # This ensures the instrumentation wraps the mock, not the real function
            with patch(
                "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call"
            ) as mock_execute_tool:
                mock_result = ContinueFlowWithNextStep(events=[])
                mock_execute_tool.return_value = mock_result

                instrumentation.instrument(tracer_provider)

                module = importlib.import_module(
                    "rasa.core.policies.flows.mcp_tool_executor"
                )
                call_mcp_tool = getattr(module, "call_mcp_tool")

                initial_events: List[Any] = []
                await call_mcp_tool(
                    initial_events, dialogue_stack, mcp_call_flow_step, tracker
                )

                mock_histogram.record.assert_called_once()

                call_args = mock_histogram.record.call_args
                assert "amount" in call_args.kwargs
                assert "attributes" in call_args.kwargs

                attributes = call_args.kwargs["attributes"]
                assert attributes["tool_id"] == "test_tool"
                assert attributes["mcp_server"] == "test_server"
                assert attributes["success"] == "true"
                assert call_args.kwargs["amount"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value_length,expected_length,description",
    [
        (600, 500, "truncated to max length"),
        (500, 500, "unchanged at exact boundary"),
    ],
)
async def test_tool_output_value_truncation(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    mcp_call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
    value_length: int,
    expected_length: int,
    description: str,
) -> None:
    """Test that tool output slot values are properly handled at boundary conditions."""
    import json

    from rasa.core.policies.flows.flow_step_result import ContinueFlowWithNextStep
    from rasa.shared.core.events import SlotSet

    # Create test value
    test_value = "x" * value_length

    # Mock result with test value
    mock_result = ContinueFlowWithNextStep(events=[SlotSet("result_slot", test_value)])

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call",
        return_value=mock_result,
    ):
        instrumentation.instrument(tracer_provider)

        # Execute tool call
        module = importlib.import_module("rasa.core.policies.flows.mcp_tool_executor")
        call_mcp_tool = getattr(module, "call_mcp_tool")
        await call_mcp_tool([], dialogue_stack, mcp_call_flow_step, tracker)

    # Verify truncation behavior
    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans > 0

    tool_spans = [
        span
        for span in captured_spans
        if span.name
        == "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call"
    ]
    assert len(tool_spans) > 0

    tool_span = tool_spans[0]
    assert tool_span.attributes is not None
    assert "tool_output_slots" in tool_span.attributes

    # Parse the tool output slots and verify truncation behavior
    tool_output_slots = json.loads(tool_span.attributes["tool_output_slots"])
    assert len(tool_output_slots) == 1
    assert tool_output_slots[0]["slot"] == "result_slot"
    assert len(tool_output_slots[0]["value"]) == expected_length
    assert tool_output_slots[0]["value"] == "x" * expected_length


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case,initial_events,result_events,expected_event_count,expected_slots,has_error_frame",
    [
        (
            "success_with_new_events",
            [SlotSet("initial_slot", "initial_value")],
            [
                SlotSet("initial_slot", "initial_value"),
                SlotSet("tool_slot", "tool_value"),
            ],
            1,
            [{"slot": "tool_slot", "value": "tool_value"}],
            False,
        ),
        (
            "error_no_new_events",
            [SlotSet("initial_slot", "initial_value")],
            [SlotSet("initial_slot", "initial_value")],
            0,
            [],
            True,
        ),
        (
            "success_multiple_new_events",
            [SlotSet("initial_slot", "initial_value")],
            [
                SlotSet("initial_slot", "initial_value"),
                SlotSet("tool_slot1", "tool_value1"),
                SlotSet("tool_slot2", "tool_value2"),
            ],
            2,
            [
                {"slot": "tool_slot1", "value": "tool_value1"},
                {"slot": "tool_slot2", "value": "tool_value2"},
            ],
            False,
        ),
    ],
)
async def test_tool_event_processing_and_error_detection(
    test_case: str,
    initial_events: List[SlotSet],
    result_events: List[SlotSet],
    expected_event_count: int,
    expected_slots: List[dict],
    has_error_frame: bool,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    mcp_call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
) -> None:
    """Test tool event processing and error detection with parametrized test cases."""
    from rasa.core.policies.flows.flow_step_result import ContinueFlowWithNextStep
    from rasa.dialogue_understanding.patterns.internal_error import (
        InternalErrorPatternFlowStackFrame,
    )

    # Mock result with the specified events
    mock_result = ContinueFlowWithNextStep(events=result_events)

    # Add error frame to stack if this is an error test case
    if has_error_frame:
        dialogue_stack.push(InternalErrorPatternFlowStackFrame())

    with patch(
        "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call",
        return_value=mock_result,
    ):
        instrumentation.instrument(tracer_provider)

        # Execute tool call
        module = importlib.import_module("rasa.core.policies.flows.mcp_tool_executor")
        call_mcp_tool = getattr(module, "call_mcp_tool")
        await call_mcp_tool(initial_events, dialogue_stack, mcp_call_flow_step, tracker)

    # Verify spans were captured
    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans > 0

    tool_spans = [
        span
        for span in captured_spans
        if span.name
        == "rasa.core.policies.flows.mcp_tool_executor._execute_mcp_tool_call"
    ]
    assert len(tool_spans) > 0

    tool_span = tool_spans[0]
    assert tool_span.attributes is not None

    # Check event count
    assert tool_span.attributes["tool_output_events_count"] == expected_event_count

    # Check tool output slots
    if expected_slots:
        assert "tool_output_slots" in tool_span.attributes
        tool_output_slots = json.loads(tool_span.attributes["tool_output_slots"])
        assert len(tool_output_slots) == len(expected_slots)
        for i, expected_slot in enumerate(expected_slots):
            assert tool_output_slots[i]["slot"] == expected_slot["slot"]
            assert tool_output_slots[i]["value"] == expected_slot["value"]
    else:
        assert "tool_output_slots" not in tool_span.attributes
