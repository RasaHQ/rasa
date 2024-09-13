import pytest
from typing import Any, Dict, Sequence

from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockCustomActionExecutor,
    MockRetryCustomActionExecutor,
    MockNoEndpointCustomActionExecutor,
    MockHTTPCustomActionExecutor,
    MockGRPCCustomActionExecutor,
)
from rasa.utils.endpoints import EndpointConfig


def mock_action_name() -> str:
    return "test_action"


def mock_url() -> str:
    return "http://localhost:5055/webhook"


def mock_endpoint() -> EndpointConfig:
    return EndpointConfig(mock_url())


@pytest.mark.parametrize(
    "component_class, arg_dict, action_name, action_endpoint, span_name",
    [
        (MockCustomActionExecutor, {}, "None", "None", "MockCustomActionExecutor"),
        (
            MockNoEndpointCustomActionExecutor,
            {"action_name": mock_action_name()},
            mock_action_name(),
            "None",
            "MockNoEndpointCustomActionExecutor",
        ),
        (
            MockHTTPCustomActionExecutor,
            {"action_name": mock_action_name(), "action_endpoint": mock_endpoint()},
            mock_action_name(),
            mock_url(),
            "MockHTTPCustomActionExecutor",
        ),
        (
            MockGRPCCustomActionExecutor,
            {"action_name": mock_action_name(), "action_endpoint": mock_endpoint()},
            mock_action_name(),
            mock_url(),
            "MockGRPCCustomActionExecutor",
        ),
        (
            MockRetryCustomActionExecutor,
            {"custom_action_executor": MockCustomActionExecutor()},
            "None",
            "None",
            "MockRetryCustomActionExecutor",
        ),
    ],
)
async def test_tracing_custom_action_executor_run_no_endpoint(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    component_class: Any,
    arg_dict: Dict[str, Any],
    action_name: str,
    action_endpoint: Any,
    span_name: str,
) -> None:
    instrumentation.instrument(
        tracer_provider,
        custom_action_executor_subclasses=[component_class],
    )

    mock_custom_action_executor = component_class(**arg_dict)
    tracker = DialogueStateTracker.from_events("test", evts=[SlotSet("foo", "bar")])
    domain = Domain.from_dict(
        {
            "responses": {
                "utter_greet": [{"text": "Hey there! How can I help you?"}],
            }
        }
    )

    await mock_custom_action_executor.run(tracker, domain)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == f"{span_name}.run"

    expected_attributes = {
        "class_name": component_class.__name__,
        "action_name": action_name,
        "url": action_endpoint,
        "sender_id": "test",
    }
    assert captured_span.attributes == expected_attributes
