import logging
from typing import Sequence

import pytest
from pytest import LogCaptureFixture
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext
from rasa.core.channels import UserMessage

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockAgent


@pytest.mark.asyncio
async def test_tracing_for_handle_message(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        agent_class=MockAgent,
    )

    expected_channel = "channel"
    expected_sender_id = "12345"

    agent = MockAgent()

    await agent.handle_message(
        UserMessage(
            input_channel=expected_channel,
            sender_id=expected_sender_id,
        )
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockAgent.handle_message"
    assert captured_span.attributes == {
        "input_channel": expected_channel,
        "sender_id": expected_sender_id,
        "model_id": agent.model_id,
        "model_name": agent.processor.model_filename,
    }


async def test_tracing_for_handle_message_with_headers(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    caplog: LogCaptureFixture,
) -> None:
    headers = {"traceparent": "00-ec6fbe3de34342bac2e9fe6e955354cb-c04faeec2b0670e4-01"}
    instrumentation.instrument(
        tracer_provider,
        agent_class=MockAgent,
    )

    expected_channel = "channel"
    expected_sender_id = "12345"
    agent = MockAgent()
    message = UserMessage(
        input_channel=expected_channel, sender_id=expected_sender_id, headers=headers
    )

    with caplog.at_level(logging.DEBUG):
        await agent.handle_message(message)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockAgent.handle_message"
    assert captured_span.attributes == {
        "input_channel": expected_channel,
        "sender_id": expected_sender_id,
        "model_id": agent.model_id,
        "model_name": agent.processor.model_filename,
    }

    traceparent = headers["traceparent"].split("-")
    span_context: SpanContext = captured_span.get_span_context()  # type: ignore
    assert span_context.trace_id == int(traceparent[1], 16)
    assert span_context.trace_flags == int(traceparent[3])

    debug_log_message = (
        f"The trace id for the current span "
        f"'{captured_span.name}' is '{span_context.trace_id}'."
    )
    assert debug_log_message in caplog.text
