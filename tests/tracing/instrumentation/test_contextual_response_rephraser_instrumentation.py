import logging
from typing import Any, Dict, Sequence
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.shared.constants import LLM_API_HEALTH_CHECK_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import DEFAULT_OPENAI_GENERATE_MODEL_NAME, AvailableEndpoints
from rasa.tracing.instrumentation import instrumentation
from rasa.utils.endpoints import EndpointConfig
from tests.tracing.instrumentation.conftest import (
    MockAvailableEndpoints,
    MockContextualResponseRephraser,
    TestSpanExporter,
)


class IncompleteAvailableEndpoints:
    @staticmethod
    def get_instance():
        return IncompleteAvailableEndpoints()

    def __init__(self):
        self.model_groups = [{"id": "no-llm-models-group", "models": [None]}]


@pytest.fixture
def mock_endpoints_for_rephraser(monkeypatch) -> IncompleteAvailableEndpoints:
    """Fixture to mock the endpoints for the rephraser."""
    mock = IncompleteAvailableEndpoints()

    def mock_get_instance(*args, **kwargs):
        return mock

    monkeypatch.setattr(AvailableEndpoints, "get_instance", mock_get_instance)
    return mock


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
    "llm_config, mock_env_key, expected",
    [
        (
            {
                "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
            },
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
        (
            {
                "provider": "cohere",
                "model": "cohere/gptd-instruct-tft",
                "temperature": 0.7,
                "request_timeout": 10,
            },
            "COHERE_API_KEY",
            {
                "llm_type": "cohere",
                "llm_model": "cohere/gptd-instruct-tft",
                "llm_model_group_id": "None",
                "llm_temperature": "0.7",
                "llm_request_timeout": "10",
                "request_timeout": "10",
            },
        ),
        (
            {"model_name": "gpt-3.5-turbo"},
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": "gpt-3.5-turbo",
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
        (
            {},
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
    ],
)
async def test_tracing_contextual_response_rephraser_generate_llm_response(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    llm_config: Dict[str, Any],
    expected: Dict[str, Any],
    mock_env_key: str,
    monkeypatch: MonkeyPatch,
    mock_available_endpoints: MockAvailableEndpoints,
) -> None:
    monkeypatch.setenv(mock_env_key, "mock key in test_tracing_rephraser")

    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )

    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._generate_llm_response("some text")

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
        # llm attributes
        "llm_temperature": "0.3",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "None",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": "{}",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_generate_llm_response_no_model_group(  # noqa: E501
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    monkeypatch: MonkeyPatch,
    mock_endpoints_for_rephraser: IncompleteAvailableEndpoints,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test_key")
    llm_config = {"model_group": "no-llm-models-group"}
    expected = {
        "llm_model": "None",
        "llm_type": "None",
        "llm_model_group_id": "no-llm-models-group",
        "llm_temperature": "None",
        "llm_request_timeout": "None",
        "request_timeout": "None",
    }

    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )

    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._generate_llm_response("some text")

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
        # llm attributes
        "llm_temperature": "0.3",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "None",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": "{}",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_rephrase(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_tracing_rephraser")
    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
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

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

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
    domain_with_responses: Domain,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_tracing_rephraser")
    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )

    endpoint_config = EndpointConfig.from_dict({"trace_prompt_tokens": True})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._generate_llm_response("This is a test prompt.")

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )
    expected_attributes = {
        "class_name": component_class.__name__,
        "len_prompt_tokens": "6",
        # llm attributes
        "llm_type": "openai",
        "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
        "llm_model_group_id": "None",
        "llm_temperature": "0.3",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "None",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": "{}",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_len_prompt_tokens_non_openai(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("COHERE_API_KEY", "mock key in test_tracing_rephraser")
    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )
    endpoint_config = EndpointConfig.from_dict(
        {"trace_prompt_tokens": True, "llm": {"provider": "cohere", "model": "command"}}
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

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert (
        captured_span.name == "MockContextualResponseRephraser._generate_llm_response"
    )

    assert captured_span.attributes["len_prompt_tokens"] == "None"


@pytest.mark.parametrize(
    "llm_config, mock_env_key, expected",
    [
        (
            {
                "provider": "openai",
                "model_name": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
            },
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
        (
            {
                "provider": "cohere",
                "model": "cohere/gptd-instruct-tft",
                "temperature": 0.7,
                "request_timeout": 10,
            },
            "COHERE_API_KEY",
            {
                "llm_type": "cohere",
                "llm_model": "cohere/gptd-instruct-tft",
                "llm_model_group_id": "None",
                "llm_temperature": "0.7",
                "llm_request_timeout": "10",
                "request_timeout": "10",
            },
        ),
        (
            {"model_name": "gpt-3.5-turbo"},
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": "gpt-3.5-turbo",
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
        (
            {},
            OPENAI_API_KEY_ENV_VAR,
            {
                "llm_model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
                "llm_type": "openai",
                "llm_model_group_id": "None",
            },
        ),
    ],
)
async def test_tracing_contextual_response_rephraser_create_history(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
    llm_config: Dict[str, Any],
    expected: Dict[str, Any],
    mock_env_key: str,
    monkeypatch: MonkeyPatch,
    mock_available_endpoints: MockAvailableEndpoints,
) -> None:
    monkeypatch.setenv(mock_env_key, "mock key in test_tracing_rephraser")
    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )
    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._create_history(greet_tracker)

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockContextualResponseRephraser._create_history"
    expected_attributes = {
        "class_name": component_class.__name__,
        # llm attributes
        "llm_type": "openai",
        "llm_temperature": "0.3",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "None",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": "{}",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes


async def test_tracing_contextual_response_rephraser_create_history_no_model_group(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
    mock_endpoints_for_rephraser: IncompleteAvailableEndpoints,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "test_key")

    llm_config = {"model_group": "no-llm-models-group"}
    expected = {
        "llm_model": "None",
        "llm_type": "None",
        "llm_model_group_id": "no-llm-models-group",
        "llm_temperature": "None",
        "llm_request_timeout": "None",
        "request_timeout": "None",
    }

    test_span_exported = TestSpanExporter(span_exporter)
    ignore_substrings = ["health_check"]
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans(
        ignore_substrings
    )
    endpoint_config = EndpointConfig.from_dict({"llm": llm_config})
    mock_rephraser = component_class(
        endpoint_config=endpoint_config, domain=domain_with_responses
    )

    await mock_rephraser._create_history(greet_tracker)

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans(
        ignore_substrings
    )  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "MockContextualResponseRephraser._create_history"
    expected_attributes = {
        "class_name": component_class.__name__,
        # llm attributes
        "llm_type": "openai",
        "llm_temperature": "0.3",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "None",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": "{}",
    }
    expected_attributes.update(expected)
    assert captured_span.attributes == expected_attributes


@pytest.mark.parametrize(
    "mock_perform_health_check,"
    "expected_attributes,"
    "confing_present_in_expected_attributes,"
    "llm_api_health_check_env_var_value",
    [
        (
            Mock(return_value="returned-model-health-check-success"),
            {
                "api_health_check_enabled": True,
                "health_check_trigger_component": "ContextualResponseRephraser",
                "health_check_trigger_method": "contextual_response_rephraser.init",
            },
            True,
            "true",
        ),
        (
            Mock(return_value=None),
            {
                "api_health_check_enabled": True,
                "health_check_trigger_component": "ContextualResponseRephraser",
                "health_check_trigger_method": "contextual_response_rephraser.init",
            },
            True,
            "true",
        ),
        (
            Mock(),
            {
                "api_health_check_enabled": False,
                "health_check_trigger_component": "ContextualResponseRephraser",
                "health_check_trigger_method": "contextual_response_rephraser.init",
            },
            False,
            "false",
        ),
    ],
)
async def test_tracing_contextual_response_rephraser_health_check_success(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    domain_with_responses: Domain,
    greet_tracker: DialogueStateTracker,
    monkeypatch: MonkeyPatch,
    mock_perform_health_check: Mock,
    expected_attributes: Dict[str, Any],
    confing_present_in_expected_attributes: bool,
    llm_api_health_check_env_var_value: str,
    mock_perform_llm_health_check: Mock,
) -> None:
    # Given
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "mock key in test_tracing_rephraser")
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, llm_api_health_check_env_var_value)

    test_span_exported = TestSpanExporter(span_exporter)
    component_class = MockContextualResponseRephraser
    instrumentation.instrument(
        tracer_provider,
        contextual_response_rephraser_class=component_class,
    )
    previous_num_captured_spans = test_span_exported.get_previous_num_captured_spans()
    endpoint_config = EndpointConfig.from_dict({})

    # When: Health check is happening in __init__ in the ContextualResponseRephraser
    component_class(endpoint_config=endpoint_config, domain=domain_with_responses)

    captured_spans: Sequence[ReadableSpan] = test_span_exported.get_finished_spans()
    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    captured_span = captured_spans[-1]

    # Then
    assert num_captured_spans == 1
    assert captured_span.name == (
        "MockContextualResponseRephraser.perform_llm_health_check"
    )

    if confing_present_in_expected_attributes:
        assert captured_span.attributes["config"] is not None
        assert bool(captured_span.attributes["config"])
    else:
        assert "config" not in captured_span.attributes

    for key, value in expected_attributes.items():
        assert captured_span.attributes[key] == value
