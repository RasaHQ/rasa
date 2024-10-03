from typing import Sequence
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.message import Message
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockNLUCommandAdapter


@pytest.mark.parametrize(
    "predict_commands_kwargs",
    [
        {},
        {"domain": Mock(spec_set=Domain)},
    ],
)
async def test_tracing_nlu_command_adapter_predict_commands(
    predict_commands_kwargs: dict,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    default_model_storage: ModelStorage,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockNLUCommandAdapter

    instrumentation.instrument(
        tracer_provider,
        nlu_command_adapter_class=component_class,
    )

    mock_nlu_command_adapter = component_class(
        config={},
        model_storage=default_model_storage,
        resource=Resource("nlu_command_adapter"),
        execution_context=Mock(spec=ExecutionContext),
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[UserUttered("I need some medical advice.")],
    )

    await mock_nlu_command_adapter.predict_commands(
        Message(data={"intent": {"name": "health_advice"}}),
        FlowsList(underlying_flows=[]),
        tracker,
        **predict_commands_kwargs,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockNLUCommandAdapter.predict_commands"

    assert captured_span.attributes == {
        "commands": '[{"flow": "health_advice", "command": "start flow"}]',
        "intent": '"health_advice"',
    }
