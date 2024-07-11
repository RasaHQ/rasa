import inspect
from typing import Dict, Sequence, Tuple
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext
from rasa.core.actions.action import ActionBotResponse, RemoteAction
from rasa.core.channels import UserMessage
from rasa.utils.endpoints import EndpointConfig

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockMessageProcessor,
    TrackerMock,
)

EVENTS_IN_TRACKER = ["foo", "bar", "baz"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method_name, method_args, expected_attributes",
    [
        (
            "handle_message",
            (UserMessage(),),
            {},
        ),
        (
            "log_message",
            (UserMessage(),),
            {},
        ),
        (
            "get_tracker",
            ("conversation_id",),
            {"number_of_events": len(EVENTS_IN_TRACKER)},
        ),
        (
            "_run_action",
            (ActionBotResponse("I am an action"), Mock(), Mock(), Mock(), Mock()),
            {"action_name": "I am an action"},
        ),
        (
            "save_tracker",
            (TrackerMock(EVENTS_IN_TRACKER),),
            {"number_of_events": len(EVENTS_IN_TRACKER)},
        ),
        (
            "_run_prediction_loop",
            (
                Mock(),
                Mock(),
            ),
            {},
        ),
        (
            "_predict_next_with_tracker",
            (TrackerMock(EVENTS_IN_TRACKER),),
            {"intent_name": "greet", "requested_slot": "test_slot"},
        ),
    ],
)
async def test_tracing(
    method_name: str,
    method_args: Tuple,
    expected_attributes: Dict,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        processor_class=MockMessageProcessor,
    )

    processor = MockMessageProcessor(EVENTS_IN_TRACKER)

    method_to_call = getattr(processor, method_name)
    if inspect.iscoroutinefunction(method_to_call):
        await getattr(processor, method_name)(*method_args)
    else:
        getattr(processor, method_name)(*method_args)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == f"MockMessageProcessor.{method_name}"
    assert captured_span.attributes == expected_attributes


@pytest.mark.asyncio
async def test_propagation_to_action_server(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        processor_class=MockMessageProcessor,
    )

    processor = MockMessageProcessor(EVENTS_IN_TRACKER)
    action_server_url = "http://localhost:5055/webhook"
    remote_action = RemoteAction("custom_action", EndpointConfig(action_server_url))

    assert remote_action.action_endpoint.headers == {}

    await processor._run_action(remote_action, Mock(), Mock(), Mock(), Mock())

    assert "traceparent" in remote_action.action_endpoint.headers
    id_list = remote_action.action_endpoint.headers["traceparent"].split("-")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockMessageProcessor._run_action"
    assert captured_span.attributes == {
        "action_name": "custom_action",
        "executor_class_name": "HTTPCustomActionExecutor",
    }

    span_context: SpanContext = captured_span.get_span_context()  # type: ignore

    assert span_context.trace_id == int(id_list[1], 16)
    assert span_context.span_id == int(id_list[2], 16)
