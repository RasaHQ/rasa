from typing import Any, Dict, Sequence, Text

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import DEFAULT_OPENAI_GENERATE_MODEL_NAME
from rasa.utils.endpoints import EndpointConfig

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockContextualResponseRephraser


@pytest.fixture
def domain_with_responses() -> Domain:
    return Domain.from_dict(
        {
            "responses": {
                "utter_allows_rephrasing": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase": True},
                    }
                ],
                "utter_does_not_allow_rephrasing": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase": False},
                    }
                ],
                "utter_no_metadata": [{"text": "Hey there! How can I help you?"}],
                "utter_with_prompt": [
                    {
                        "text": "Hey there! How can I help you?",
                        "metadata": {"rephrase_prompt": "foobar", "rephrase": True},
                    }
                ],
            }
        }
    )


@pytest.fixture
def greet_tracker() -> DialogueStateTracker:
    return DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("Hello", {"name": "greet", "confidence": 1.0}),
            SlotSet(
                "dialogue_stack",
                [
                    {
                        "frame_id": "YE9C8M8R",
                        "flow_id": "pattern_collect_information",
                        "step_id": "ask_collect",
                        "collect": "confirm_slot_correction",
                        "utter": "utter_ask_confirm_slot_correction",
                        "rejections": [],
                        "type": "pattern_collect_information",
                    },
                ],
            ),
        ],
    )


@pytest.mark.parametrize(
    "llm_config, expected",
    [
        (
            {
                "_type": "test",
                "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
            },
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "test",
            },
        ),
        (
            {
                "_type": "test",
                "model": None,
            },
            {
                "llm_type": "test",
            },
        ),
        (
            {
                "_type": "test",
                "model_name": None,
            },
            {
                "llm_type": "test",
            },
        ),
        (
            {"model_name": "gpt-3.5-turbo"},
            {
                "llm_model": "gpt-3.5-turbo",
                "llm_type": "openai",
            },
        ),
        (
            {},
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_tracing_contextual_response_rephrasal(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    llm_config: Dict[str, Any],
    expected: Text,
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_command = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    mock_command._generate_llm_response("some text")

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
    }
    expected_attributes.update(expected)  # type: ignore[arg-type]
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephrasal_rephrase(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict({})
    mock_command = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_command.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockContextualResponseRephraser.generate"

    expected_attributes = {
        "class_name": component_class.__name__,
        "utter": "utter_allows_rephrasing",
    }
    assert captured_span.attributes == expected_attributes
