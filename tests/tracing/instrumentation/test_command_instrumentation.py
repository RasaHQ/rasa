from typing import Sequence

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockCommand


@pytest.mark.asyncio
async def test_tracing_command_run_command_on_tracker(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockCommand

    instrumentation.instrument(
        tracer_provider,
        command_subclasses=[component_class],  # type: ignore[arg-type]
    )

    mock_command = component_class()
    tracker = DialogueStateTracker.from_events("test", evts=[SlotSet("foo", "bar")])
    mock_command.run_command_on_tracker(
        tracker, FlowsList(underlying_flows=[]), tracker
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockCommand.run_command_on_tracker"

    expected_attributes = {
        "class_name": component_class.__name__,
        "number_of_events": 1,
        "sender_id": "test",
    }
    assert captured_span.attributes == expected_attributes
