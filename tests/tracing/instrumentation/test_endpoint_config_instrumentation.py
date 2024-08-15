import json
from typing import Sequence

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanContext

from rasa.tracing.constants import REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockEndpointConfig


async def test_tracing_endpoint_config_request(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockEndpointConfig

    instrumentation.instrument(
        tracer_provider,
        endpoint_config_class=component_class,
    )

    test_url = "http://localhost:5055/webhook"
    mock_endpoint_config = component_class(url=test_url)

    test_json = {"test": "value"}
    await mock_endpoint_config.request(json=test_json)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockEndpointConfig.request"

    expected_attributes = {
        "url": test_url,
        REQUEST_BODY_SIZE_IN_BYTES_ATTRIBUTE_NAME: len(
            json.dumps(test_json).encode("utf-8")
        ),
    }
    assert captured_span.attributes == expected_attributes

    assert "traceparent" in mock_endpoint_config.headers
    id_list = mock_endpoint_config.headers["traceparent"].split("-")
    span_context: SpanContext = captured_span.get_span_context()  # type: ignore

    assert span_context.trace_id == int(id_list[1], 16)
