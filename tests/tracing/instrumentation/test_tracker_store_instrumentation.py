from typing import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.shared.core.events import ActionExecuted

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockEventBroker,
    MockTrackerStore,
)

NEW_EVENTS_IN_TRACKER = [
    ActionExecuted("action_listen"),
    ActionExecuted("utter_greet"),
    ActionExecuted("utter_bye"),
]


@pytest.mark.asyncio
async def test_tracing_for_stream_events(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        tracker_store_class=MockTrackerStore,
    )

    expected_number_of_events = len(NEW_EVENTS_IN_TRACKER)
    expected_broker_class = MockEventBroker

    event_broker = MockEventBroker()
    tracker_store = MockTrackerStore(event_broker)
    await tracker_store._stream_new_events(event_broker, NEW_EVENTS_IN_TRACKER, "test")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockTrackerStore._stream_new_events"
    assert captured_span.attributes == {
        "number_of_streamed_events": expected_number_of_events,
        "broker_class": expected_broker_class.__name__,
    }
