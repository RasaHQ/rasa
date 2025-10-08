import importlib
import json
from typing import Any, List, Sequence
from unittest.mock import MagicMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.events import SessionStarted
from rasa.shared.core.flows.flow_step_links import FlowStepLinks
from rasa.shared.core.flows.flows_list import FlowsList
from rasa.shared.core.flows.steps import CallFlowStep
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.tracing.constants import AGENT_EXECUTION_DURATION_METRIC_NAME
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.instrumentation import (
    AGENT_EXECUTOR_MODULE_NAME,
    FLOW_EXECUTOR_MODULE_NAME,
)
from rasa.tracing.metric_instrument_provider import MetricInstrumentProvider
from tests.tracing.instrumentation.conftest import MockAgentWithToolCall


@pytest.fixture
def tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events("test", evts=[SessionStarted()])


@pytest.fixture
def slots() -> List[Slot]:
    return []


@pytest.fixture
def flows() -> FlowsList:
    return FlowsList(underlying_flows=[])


@pytest.fixture
def dialogue_stack() -> DialogueStack:
    return DialogueStack(frames=[])


@pytest.fixture
def call_flow_step() -> CallFlowStep:
    """Create a CallFlowStep for testing."""
    return CallFlowStep(
        idx=1,
        custom_id="test_call_step",
        description="Test call step",
        call="test_agent",
        flow_id="test_flow",
        metadata={},
        next=FlowStepLinks(links=[]),
    )


@pytest.mark.asyncio
async def test_tracing_agent_execution(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    tracker: DialogueStateTracker,
    call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
    slots: List[Slot],
    flows: FlowsList,
) -> None:
    """Test tracing for agent execution in flow steps."""
    with patch(
        f"{AGENT_EXECUTOR_MODULE_NAME}._call_agent_with_retry"
    ) as mock_call_agent:
        mock_result = MagicMock()
        mock_result.events = []
        mock_result.status = "success"
        mock_result.id = "test_agent_response"
        mock_result.response_message = "Test response"
        mock_result.structured_results = []
        mock_result.error_message = None
        mock_result.metadata = {"test": "metadata"}
        mock_call_agent.return_value = mock_result

        instrumentation.instrument(tracer_provider)

        module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)
        run_agent = getattr(module, "run_agent")

        initial_events: List[Any] = []
        await run_agent(
            initial_events, dialogue_stack, call_flow_step, tracker, slots, flows
        )

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore
        num_captured_spans = len(captured_spans) - previous_num_captured_spans

        assert num_captured_spans > 0

        agent_spans = [
            span
            for span in captured_spans
            if span.name == f"{AGENT_EXECUTOR_MODULE_NAME}._call_agent_with_retry"
        ]
        assert len(agent_spans) > 0

        agent_span = agent_spans[0]
        assert agent_span.attributes is not None
        assert "agent_name" in agent_span.attributes

        agent_span = agent_spans[-1]
        assert agent_span.attributes["agent_name"] == "test_agent"

        instrument_provider = MetricInstrumentProvider()
        if instrument_provider.instruments:
            agent_metric = instrument_provider.get_instrument(
                AGENT_EXECUTION_DURATION_METRIC_NAME
            )
            assert agent_metric is not None


@pytest.mark.asyncio
async def test_histogram_recording_agent_execution(
    tracer_provider: TracerProvider,
    tracker: DialogueStateTracker,
    call_flow_step: CallFlowStep,
    dialogue_stack: DialogueStack,
    slots: List[Slot],
    flows: FlowsList,
) -> None:
    """Test that histogram recording actually works for agent execution."""
    # Mock the histogram to verify record() is called
    mock_histogram = Mock()

    # Mock the MetricInstrumentProvider to return our mock histogram
    with patch.object(
        MetricInstrumentProvider, "get_instrument", return_value=mock_histogram
    ):
        with patch.object(
            MetricInstrumentProvider, "instruments", {"test": "instruments"}
        ):
            with patch(
                f"{AGENT_EXECUTOR_MODULE_NAME}._call_agent_with_retry"
            ) as mock_call_agent:
                instrumentation.instrument(tracer_provider)

                module = importlib.import_module(FLOW_EXECUTOR_MODULE_NAME)
                run_agent = getattr(module, "run_agent")
                mock_result = Mock()
                mock_result.events = []
                mock_result.status = "success"
                mock_result.id = "test_agent_response"
                mock_result.response_message = "Test response"
                mock_result.structured_results = []
                mock_result.error_message = None
                mock_result.metadata = {"test": "metadata"}
                mock_call_agent.return_value = mock_result

                initial_events: List[Any] = []
                await run_agent(
                    initial_events,
                    dialogue_stack,
                    call_flow_step,
                    tracker,
                    slots,
                    flows,
                )

                # Check that histogram was called
                # (may be called multiple times due to state transitions)
                assert mock_histogram.record.call_count >= 1

                # Find the agent execution call (not state transition calls)
                agent_execution_calls = [
                    call
                    for call in mock_histogram.record.call_args_list
                    if call.kwargs["amount"] > 0  # Agent execution has duration > 0
                ]
                assert len(agent_execution_calls) == 1

                call_args = agent_execution_calls[0]
                assert "amount" in call_args.kwargs
                assert "attributes" in call_args.kwargs

                attributes = call_args.kwargs["attributes"]
                assert attributes["agent_name"] == "test_agent"
                assert "ProtocolType.MCP_OPEN" in attributes["protocol_type"]
                assert attributes["status"] == "success"
                assert call_args.kwargs["amount"] > 0


@pytest.mark.asyncio
async def test_tracing_agent_internal_state_transitions(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    """Test tracing for agent internal state transitions."""
    from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
        AgentStackFrame,
        AgentState,
    )

    instrumentation.instrument(tracer_provider)

    # Create an AgentStackFrame and change its state to trigger transitions
    agent_frame = AgentStackFrame(
        agent_id="test_agent",
        flow_id="test_flow",
        step_id="test_step",
        state=AgentState.WAITING_FOR_INPUT,
    )

    # Change state to INTERRUPTED
    agent_frame.state = AgentState.INTERRUPTED

    # Change state back to WAITING_FOR_INPUT
    agent_frame.state = AgentState.WAITING_FOR_INPUT

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans >= 2

    # Find agent internal state transition spans
    state_transition_spans = [
        span
        for span in captured_spans
        if span.name == "AgentStackFrame.state_transition"
    ]
    assert len(state_transition_spans) >= 2

    # Check first transition: WAITING_FOR_INPUT -> INTERRUPTED
    interrupted_span = next(
        (
            span
            for span in state_transition_spans
            if span.attributes["from_state"] == "waiting_for_input"
            and span.attributes["to_state"] == "interrupted"
        ),
        None,
    )
    assert interrupted_span is not None
    assert interrupted_span.attributes is not None
    assert interrupted_span.attributes["agent_id"] == "test_agent"
    assert interrupted_span.attributes["flow_id"] == "test_flow"
    assert interrupted_span.attributes["step_id"] == "test_step"
    assert interrupted_span.attributes["from_state"] == "waiting_for_input"
    assert interrupted_span.attributes["to_state"] == "interrupted"

    # Check second transition: INTERRUPTED -> WAITING_FOR_INPUT
    resumed_span = next(
        (
            span
            for span in state_transition_spans
            if span.attributes["from_state"] == "interrupted"
            and span.attributes["to_state"] == "waiting_for_input"
        ),
        None,
    )
    assert resumed_span is not None
    assert resumed_span.attributes is not None
    assert resumed_span.attributes["agent_id"] == "test_agent"
    assert resumed_span.attributes["flow_id"] == "test_flow"
    assert resumed_span.attributes["step_id"] == "test_step"
    assert resumed_span.attributes["from_state"] == "interrupted"
    assert resumed_span.attributes["to_state"] == "waiting_for_input"


@pytest.mark.asyncio
async def test_tracing_agent_tool_call_execution(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    """Test tracing for agent-level tool call execution."""
    instrumentation.instrument(
        tracer_provider, subagent_classes=[MockAgentWithToolCall]
    )

    agent = MockAgentWithToolCall()
    await agent._execute_tool_call("test_tool", {"param": "value"})

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans > 0

    # Find the tool call span
    tool_call_span = next(
        (
            span
            for span in captured_spans[-num_captured_spans:]
            if span.name == "MockAgentWithToolCall._execute_tool_call"
        ),
        None,
    )

    assert tool_call_span is not None
    assert tool_call_span.attributes is not None

    # Check input attributes
    assert tool_call_span.attributes["tool_name"] == "test_tool"
    assert json.loads(tool_call_span.attributes["tool_arguments"]) == {"param": "value"}
    assert tool_call_span.attributes["agent_name"] == "test_agent"
    assert tool_call_span.attributes["protocol_type"] == "ProtocolType.MCP_OPEN"
    assert tool_call_span.attributes["execution_context"] == "agent"

    # Check output attributes
    assert tool_call_span.attributes["tool_result_name"] == "test_tool"
    assert tool_call_span.attributes["tool_result_is_error"] is False
    assert tool_call_span.attributes["tool_result_content"] == "Result for test_tool"

    # Check execution time is recorded
    assert "tool_execution_duration_ns" in tool_call_span.attributes
    assert tool_call_span.attributes["tool_execution_duration_ns"] > 0
