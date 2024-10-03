from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.events import SlotSet
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockPolicy


def test_tracing_policy_prediction(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    component_class = MockPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    mock_policy = component_class(
        config={},
        model_storage=default_model_storage,
        resource=None,
        execution_context=default_execution_context,
    )

    action_metadata = {
        "message": {
            "text": "FinX offers a seamless booking experience for "
            "restaurants, hotels, and flights. "
            "You can browse the available options and make "
            "your reservations with ease. "
            "Whether you're looking to book a table at a restaurant, "
            "find the perfect hotel for your getaway, or plan your "
            "dream vacation with flights, FinX has got you covered. "
            "No need to switch between multiple apps – "
            "it's all right here at your fingertips.",
        }
    }
    mock_policy._prediction(
        probabilities=[1],
        events=[SlotSet("destination", "Berlin")],
        optional_events=[],
        diagnostic_data={},
        action_metadata=action_metadata,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockPolicy._prediction"

    expected_attributes = {
        "priority": 1,
        "events": ("SlotSet",),
        "optional_events": "None",
        "is_end_to_end_prediction": False,
        "is_no_user_prediction": False,
        "diagnostic_data": "{}",
        "action_metadata": '{"message": {"text": "FinX offers a seamless booking'
        " experience for restaurants, hotels, and flights. "
        "You can browse the available options and make your "
        "reservations with ease. Whether you're looking to "
        "book a table at a restaurant, find the perfect hotel "
        "for your getaway, or plan your dream vacation with "
        "flights, FinX has got you covered. "
        "No need to switch between multiple apps \\u2013 it's "
        'all right here at your fingertips."}}',
    }

    assert captured_span.attributes == expected_attributes
