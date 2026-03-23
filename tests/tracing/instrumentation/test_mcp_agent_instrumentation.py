"""Tests for MCP agent LLM instrumentation - consolidated version."""

import json
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from _pytest.fixtures import FixtureRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.agents.core.types import AgentStatus, ProtocolType
from rasa.agents.protocol.mcp.mcp_base_agent import DEFAULT_LLM_CONFIG
from rasa.agents.schemas import AgentInput, AgentInputSlot, AgentOutput, AgentToolSchema
from rasa.core.channels import OutputChannel
from rasa.tracing.constants import (
    AGENT_NAME_ATTRIBUTE_NAME,
    EXECUTION_CONTEXT_ATTRIBUTE_NAME,
    PROTOCOL_TYPE_ATTRIBUTE_NAME,
)
from rasa.tracing.instrumentation import instrumentation
from rasa.tracing.instrumentation.attribute_extractors import (
    extract_attrs_for_mcp_agent_llm_call,
)
from tests.tracing.instrumentation.conftest import MockMCPOpenAgent


@pytest.fixture
def mock_llm_config() -> Dict[str, str]:
    """Standard LLM config mock for testing."""
    return {
        "llm_model": "gpt-4",
        "llm_type": "openai",
        "llm_temperature": "0.7",
    }


@pytest.fixture
def sample_messages() -> List[Dict[str, str]]:
    """Standard message structure for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, help me with a task"},
    ]


@pytest.fixture
def extended_messages() -> List[Dict[str, str]]:
    """Extended message structure with assistant response for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, help me with a task"},
        {"role": "assistant", "content": "I can help you with that."},
    ]


@pytest.fixture
def mock_agent_with_config() -> Mock:
    """Pre-configured agent mock for testing."""
    agent: Mock = Mock()
    agent.__class__.__name__ = "MCPOpenAgent"
    agent.llm_client.config = DEFAULT_LLM_CONFIG
    agent._name = "mock_open_agent"
    agent.protocol_type = ProtocolType.MCP_OPEN
    return agent


@pytest.fixture
def sample_agent_input() -> AgentInput:
    """Sample agent input for testing."""
    return AgentInput(
        id="test_input_123",
        user_message="Hello, help me with a task",
        slots=[AgentInputSlot(name="user_name", value="John", type="text")],
        conversation_history="",
        events=[],
        metadata={},
    )


@pytest.fixture
def empty_agent_input() -> AgentInput:
    """Empty agent input for testing edge cases."""
    return AgentInput(
        id="empty_input",
        user_message="",
        slots=[],
        conversation_history="",
        events=[],
        metadata={},
    )


@pytest.mark.parametrize(
    "agent_type,agent_input_fixture,llm_config,expected_config",
    [
        ("MCPOpenAgent", "sample_agent_input", DEFAULT_LLM_CONFIG, DEFAULT_LLM_CONFIG),
        ("MCPTaskAgent", "empty_agent_input", DEFAULT_LLM_CONFIG, DEFAULT_LLM_CONFIG),
        ("MCPOpenAgent", "sample_agent_input", None, DEFAULT_LLM_CONFIG),
    ],
)
def test_extract_attrs_basic_functionality(
    agent_type: str,
    agent_input_fixture: str,
    llm_config: dict,
    expected_config: dict,
    request: FixtureRequest,
    mock_llm_config: Dict[str, str],
    sample_messages: List[Dict[str, str]],
) -> None:
    """Test basic attribute extraction functionality for MCP agents.

    Tests various combinations of agent types, input types, and LLM configurations.
    """
    agent: Mock = Mock()
    agent.__class__.__name__ = agent_type
    agent.llm_config = llm_config
    agent.llm_client.config = expected_config
    agent._name = "parametrized_mcp_agent"
    agent.protocol_type = (
        ProtocolType.MCP_TASK if "Task" in agent_type else ProtocolType.MCP_OPEN
    )
    agent_input = request.getfixturevalue(agent_input_fixture)

    with patch(
        "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
    ) as mock_extract_llm:
        mock_extract_llm.return_value = mock_llm_config
        agent.build_messages_for_llm_request.return_value = sample_messages
        result = extract_attrs_for_mcp_agent_llm_call(agent, agent_input)

    assert "llm_model" in result
    assert "llm_temperature" in result
    assert "llm_type" in result
    assert result[AGENT_NAME_ATTRIBUTE_NAME] == "parametrized_mcp_agent"
    assert result[EXECUTION_CONTEXT_ATTRIBUTE_NAME] == "agent"
    assert result[PROTOCOL_TYPE_ATTRIBUTE_NAME] == str(agent.protocol_type)
    mock_extract_llm.assert_called_once_with(agent, default_llm_config=expected_config)


@pytest.mark.parametrize(
    "messages_fixture,expected_token_counts,expected_total_tokens,expected_message_count",
    [
        ("sample_messages", [8, 12], "20", 2),
        ("extended_messages", [8, 12, 10], "30", 3),
    ],
)
def test_prompt_and_token_attributes(
    sample_agent_input: AgentInput,
    mock_agent_with_config: Mock,
    mock_llm_config: Dict[str, str],
    messages_fixture: str,
    expected_token_counts: List[int],
    expected_total_tokens: str,
    expected_message_count: int,
    request: FixtureRequest,
) -> None:
    """Test prompt messages structure and token counting functionality."""
    agent = mock_agent_with_config
    agent_input = sample_agent_input
    messages = request.getfixturevalue(messages_fixture)
    agent.build_messages_for_llm_request.return_value = messages

    with (
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.extract_llm_config"
        ) as mock_extract_llm,
        patch(
            "rasa.tracing.instrumentation.attribute_extractors.compute_prompt_tokens_length"
        ) as mock_compute_tokens,
    ):
        mock_extract_llm.return_value = mock_llm_config
        mock_compute_tokens.side_effect = expected_token_counts
        result = extract_attrs_for_mcp_agent_llm_call(agent, agent_input)

    assert "llm_model" in result
    assert "llm_temperature" in result
    assert "llm_type" in result
    assert "prompt_messages_count" in result
    assert result["prompt_messages_count"] == expected_message_count
    assert "len_prompt_tokens" in result
    assert result["len_prompt_tokens"] == expected_total_tokens
    assert result[AGENT_NAME_ATTRIBUTE_NAME] == "mock_open_agent"
    assert result[EXECUTION_CONTEXT_ATTRIBUTE_NAME] == "agent"
    assert result[PROTOCOL_TYPE_ATTRIBUTE_NAME] == str(ProtocolType.MCP_OPEN)
    # Note: prompt_messages content is not traced to avoid PII issues
    assert mock_compute_tokens.call_count == len(expected_token_counts)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_output,expected_response_message,expected_status,expected_events_count,should_have_response_attrs",
    [
        (
            {
                "id": "test_output_123",
                "response_message": "I can help you with that task!",
                "status": "completed",
                "events": [],
            },
            "I can help you with that task!",
            "completed",
            0,
            True,
        ),
        (
            {
                "id": "test_output_with_events",
                "response_message": "Task completed successfully!",
                "status": "completed",
                "events": ["event1", "event2", "event3"],
            },
            "Task completed successfully!",
            "completed",
            3,
            True,
        ),
        (
            None,
            None,
            None,
            None,
            False,
        ),
    ],
)
async def test_agent_llm_response_capture_in_span(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    sample_agent_input: AgentInput,
    previous_num_captured_spans: int,
    agent_output: dict,
    expected_response_message: str,
    expected_status: str,
    expected_events_count: int,
    should_have_response_attrs: bool,
) -> None:
    """Test that agent LLM response content is captured in the tracing span.

    This test verifies that the LLM response capture instrumentation works
    correctly and captures the agent output in the same span as the technical
    LLM call details, following the IntentlessPolicy pattern.
    """
    instrumentation.instrument(
        tracer_provider,
        subagent_classes=[MockMCPOpenAgent],
    )

    mock_agent = MockMCPOpenAgent()

    if agent_output is not None:
        from rasa.shared.core.events import SlotSet

        events = [
            SlotSet(key="test_slot", value=event) for event in agent_output["events"]
        ]
        mock_agent_output = AgentOutput(
            id=agent_output["id"],
            response_message=agent_output["response_message"],
            status=AgentStatus.COMPLETED,
            events=events,
        )
    else:
        mock_agent_output = None

    mock_agent.set_agent_output(mock_agent_output)
    await mock_agent.send_message(sample_agent_input)

    captured_spans = span_exporter.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    if should_have_response_attrs:
        assert num_captured_spans == 2
    else:
        assert num_captured_spans == 1

    technical_span = None
    response_span = None

    expected_spans = 2 if should_have_response_attrs else 1
    for span in captured_spans[-expected_spans:]:
        if span.name == "MockMCPOpenAgent.send_message":
            technical_span = span
        elif span.name == "MockMCPOpenAgent.send_message.llm_response":
            response_span = span

    assert technical_span is not None, "Technical LLM call span not found"
    assert "llm_model" in technical_span.attributes
    assert "llm_type" in technical_span.attributes
    assert "prompt_messages_count" in technical_span.attributes
    assert technical_span.attributes[AGENT_NAME_ATTRIBUTE_NAME] == "MockMCPOpenAgent"
    assert technical_span.attributes[EXECUTION_CONTEXT_ATTRIBUTE_NAME] == "agent"
    assert technical_span.attributes[PROTOCOL_TYPE_ATTRIBUTE_NAME] == str(
        ProtocolType.MCP_OPEN
    )

    if should_have_response_attrs:
        assert response_span is not None, "Response capture span not found"
        assert "agent_output_response_message" in response_span.attributes
        assert "agent_output_id" in response_span.attributes
        assert "agent_output_status" in response_span.attributes
        assert "agent_output_events_count" in response_span.attributes

        assert (
            response_span.attributes["agent_output_response_message"]
            == expected_response_message
        )
        assert response_span.attributes["agent_output_id"] == agent_output["id"]
        assert response_span.attributes["agent_output_status"] == expected_status
        assert (
            response_span.attributes["agent_output_events_count"]
            == expected_events_count
        )
    else:
        assert (
            response_span is None
        ), "Response span should not be created when output is None"


@pytest.mark.asyncio
async def test_get_available_tools_tracing(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    sample_agent_input: AgentInput,
    previous_num_captured_spans: int,
) -> None:
    """Test that get_available_tools method is properly traced."""
    instrumentation.instrument(tracer_provider, subagent_classes=[MockMCPOpenAgent])

    # Setup mock agent with tools
    mock_agent = MockMCPOpenAgent()
    mock_agent._name = "test_mcp_agent"
    mock_agent._mcp_tools = [
        AgentToolSchema(
            name="mcp_tool",
            description="MCP tool",
            parameters={"type": "object", "properties": {}},
            strict=False,
        )
    ]
    mock_agent.get_agent_specific_built_in_tools = lambda _: [
        AgentToolSchema(
            name="built_in_tool",
            description="Built-in tool",
            parameters={"type": "object", "properties": {}},
            strict=False,
        )
    ]
    mock_agent.get_custom_tools = lambda: []

    # Call method and verify span creation
    mock_agent.get_available_tools(sample_agent_input)

    captured_spans = span_exporter.get_finished_spans()
    assert len(captured_spans) - previous_num_captured_spans == 1

    tools_span = captured_spans[-1]
    assert tools_span.name == "MockMCPOpenAgent.get_available_tools"
    assert tools_span.attributes[AGENT_NAME_ATTRIBUTE_NAME] == "test_mcp_agent"
    assert tools_span.attributes["total_available_tools_count"] == 2

    # Verify tools JSON structure
    tools_dict = json.loads(tools_span.attributes["available_tools"])
    assert tools_dict == {"mcp_tool": "MCP tool", "built_in_tool": "Built-in tool"}


class _RecordOutputChannelMCPAgent(MockMCPOpenAgent):
    """MCP agent that records the output_channel passed to send_message.

    Used to verify the tracing wrapper forwards output_channel to the underlying
    implementation (fix for "No output channel or recipient ID provided" when
    tracing is enabled).
    """

    def __init__(self) -> None:
        super().__init__()
        self.received_output_channel: Optional[OutputChannel] = None

    async def send_message(
        self, agent_input: AgentInput, output_channel: Optional[OutputChannel] = None
    ) -> AgentOutput:
        self.received_output_channel = output_channel
        return await super().send_message(agent_input, output_channel)


@pytest.mark.asyncio
async def test_send_message_response_capture_wrapper_forwards_output_channel(
    tracer_provider: TracerProvider,
    sample_agent_input: AgentInput,
) -> None:
    """Test that the send_message response-capture wrapper forwards output_channel.

    When tracing is enabled, send_message is wrapped to capture the LLM response
    in a span. The wrapper must pass output_channel through to the underlying
    implementation; otherwise MCP agents raise 'No output channel or recipient
    ID provided' when generating responses.
    """
    instrumentation.instrument(
        tracer_provider,
        subagent_classes=[_RecordOutputChannelMCPAgent],
    )

    mock_output_channel = Mock(spec=OutputChannel)
    agent = _RecordOutputChannelMCPAgent()
    agent.set_agent_output(
        AgentOutput(
            id=sample_agent_input.id,
            status=AgentStatus.COMPLETED,
            response_message="OK",
            events=[],
        )
    )

    await agent.send_message(sample_agent_input, output_channel=mock_output_channel)

    assert agent.received_output_channel is mock_output_channel
