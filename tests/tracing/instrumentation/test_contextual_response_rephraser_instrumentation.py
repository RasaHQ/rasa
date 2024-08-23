import logging
from typing import Any, Dict, Sequence

import pytest
from pytest import LogCaptureFixture
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
                "type": "openai",
                "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
            },
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
            },
        ),
        (
            {
                "model": "cohere/gptd-instruct-tft",
                "temperature": 0.7,
                "request_timeout": 10,
            },
            {
                "llm_type": "cohere",
                "llm_model": "cohere/gptd-instruct-tft",
                "llm_temperature": "0.7",
                "request_timeout": "10",
            },
        ),
        (  # Invalid type, should default to the type from the default config.
            {
                "type": "test",
                "model_name": None,
                "temperature": 0.7,
                "request_timeout": 10,
            },
            {
                "llm_type": "openai",
                "llm_model": "gpt-3.5-turbo",
                "llm_temperature": "0.7",
                "request_timeout": "10",
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
async def test_tracing_contextual_response_rephraser_generate_llm_response(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    llm_config: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._generate_llm_response("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
        "embeddings": "{}",
        "llm_temperature": "0.3",
        "request_timeout": "5",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_rephrase(
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
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser.generate(
        "utter_allows_rephrasing",
        greet_tracker,
        output_channel="callback",
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockContextualResponseRephraser.generate"

    expected_attributes = {
        "class_name": component_class.__name__,
        "utter": "utter_allows_rephrasing",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_len_prompt_tokens(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict({"trace_prompt_tokens": True})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._generate_llm_response("This is a test prompt.")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
        "embeddings": "{}",
        "llm_temperature": "0.3",
        "request_timeout": "5",
        "len_prompt_tokens": "6",
        "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
        "llm_type": "openai",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_len_prompt_tokens_non_openai(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    caplog: LogCaptureFixture,
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict(
        {"trace_prompt_tokens": True, "llm": {"type": "cohere", "model": "command"}}
    )
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    with caplog.at_level(logging.WARNING):
        await mock_rephraser._generate_llm_response("This is a test prompt.")
        assert (
            "Tracing prompt tokens is only supported for OpenAI models. Skipping."
            in caplog.text
        )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    assert captured_span.attributes["len_prompt_tokens"] == "None"


@pytest.mark.parametrize(
    "llm_config, expected",
    [
        (
            {
                "type": "openai",
                "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
            },
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
            },
        ),
        (
            {
                "type": "cohere",
                "model": "cohere/gptd-instruct-tft",
                "temperature": 0.7,
                "request_timeout": 10,
            },
            {
                "llm_type": "cohere",
                "llm_model": "cohere/gptd-instruct-tft",
                "llm_temperature": "0.7",
                "request_timeout": "10",
            },
        ),
        (  # Invalid type, should default to the type from the default config.
            {
                "type": "test",
                "model_name": None,
                "temperature": 0.7,
                "request_timeout": 10,
            },
            {
                "llm_type": "openai",
                "llm_model": "gpt-3.5-turbo",
                "llm_temperature": "0.7",
                "request_timeout": "10",
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
async def test_tracing_contextual_response_rephraser_create_history(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
    llm_config: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    component_class = MockContextualResponseRephraser

    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )

    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._create_history(greet_tracker)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockContextualResponseRephraser._create_history"

    expected_attributes = {
        "class_name": component_class.__name__,
        "embeddings": "{}",
        "llm_temperature": "0.3",
        "request_timeout": "5",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes
