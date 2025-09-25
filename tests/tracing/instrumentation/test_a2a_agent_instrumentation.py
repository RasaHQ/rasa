"""Tests for A2A agent instrumentation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.agents.protocol.a2a.a2a_agent import A2AAgent
from rasa.tracing.instrumentation import instrumentation


@pytest.mark.asyncio
async def test_a2a_agent_health_check_tracing(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    """Test that A2A agent health check method is properly traced."""
    instrumentation.instrument(tracer_provider, subagent_classes=[A2AAgent])

    # Create a mock A2A agent
    mock_agent = A2AAgent(
        name="test_a2a_agent",
        description="A test A2A agent",
        agent_card_path="test_agent_card.json",
        timeout=30,
        max_retries=3,
    )

    # Mock the agent card
    mock_agent_card = MagicMock()
    mock_agent_card.url = "http://test-agent.example.com"
    mock_agent.agent_card = mock_agent_card

    # Mock the client to raise an exception (simpler than mocking async iterator)
    mock_client = AsyncMock()
    mock_client.send_message.side_effect = Exception("Mock connection error")
    mock_agent._client = mock_client

    # Call the health check method (will raise exception, but tracing should still work)
    with pytest.raises(Exception):
        await mock_agent._perform_health_check()

    # Check that tracing span was created
    captured_spans = span_exporter.get_finished_spans()
    assert len(captured_spans) - previous_num_captured_spans == 1

    health_check_span = captured_spans[-1]
    assert health_check_span.name == "A2AAgent._perform_health_check"

    # Verify the span attributes
    assert health_check_span.attributes["health_check_trigger_component"] == "A2AAgent"
    assert (
        health_check_span.attributes["health_check_trigger_method"]
        == "_perform_health_check"
    )
    assert health_check_span.attributes["health_check_type"] == "a2a_agent_connectivity"
    assert "api_health_check_enabled" in health_check_span.attributes


@pytest.mark.asyncio
async def test_a2a_agent_health_check_tracing_without_agent_card(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    """Test A2A agent health check tracing when agent card is not available."""
    instrumentation.instrument(tracer_provider, subagent_classes=[A2AAgent])

    # Create a mock A2A agent without agent card
    mock_agent = A2AAgent(
        name="test_a2a_agent",
        description="A test A2A agent",
        agent_card_path="test_agent_card.json",
        timeout=30,
        max_retries=3,
    )
    mock_agent.agent_card = None
    mock_agent._client = None

    # Call the health check method (should raise an exception)
    with pytest.raises(Exception):
        await mock_agent._perform_health_check()

    # Check that tracing span was created even for failed health checks
    captured_spans = span_exporter.get_finished_spans()
    assert len(captured_spans) - previous_num_captured_spans == 1

    health_check_span = captured_spans[-1]
    assert health_check_span.name == "A2AAgent._perform_health_check"

    # Verify the span attributes
    assert health_check_span.attributes["health_check_trigger_component"] == "A2AAgent"
    assert (
        health_check_span.attributes["health_check_trigger_method"]
        == "_perform_health_check"
    )
    assert health_check_span.attributes["health_check_type"] == "a2a_agent_connectivity"
    assert "api_health_check_enabled" in health_check_span.attributes
