import logging
from typing import Any, Dict, Sequence, Text

import pytest
from pytest import LogCaptureFixture
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.dialogue_understanding.generator import MultiStepLLMCommandGenerator
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.conftest import TRACING_TESTS_FIXTURES_DIRECTORY
from tests.tracing.instrumentation.conftest import MockMultiStepLLMCommandGenerator

TEST_PROMPT_DIRECTORY = str(TRACING_TESTS_FIXTURES_DIRECTORY / "test_prompt.jinja2")


@pytest.mark.parametrize(
    "config, model",
    [
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "model_name": "gpt-4",
                    "request_timeout": 7,
                    "temperature": 0.0,
                },
            },
            "gpt-4",
        ),
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "model": "gpt-3.5-turbo",
                },
            },
            "gpt-3.5-turbo",
        ),
        ({"prompt": TEST_PROMPT_DIRECTORY}, "gpt-4"),
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "request_timeout": 7,
                    "temperature": 0.0,
                },
            },
            "gpt-4",
        ),
    ],
)
async def test_tracing_multi_step_llm_command_generator_default_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    config: Dict[str, Any],
    model: Text,
) -> None:
    component_class = MockMultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    mock_multi_step_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_multi_step_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockMultiStepLLMCommandGenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
        "llm_type": "openai",
        "llm_temperature": "0.0",
        "request_timeout": "7",
        "embeddings": "{}",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_multi_step_llm_command_generator_azure_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockMultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    model = "gpt-4"
    config = {
        "prompt": TEST_PROMPT_DIRECTORY,
        "llm": {
            "model_name": model,
            "request_timeout": 15,
            "temperature": 0.3,
            "engine": "azure-test",
        },
        "embeddings": {"deployment": "test"},
    }

    mock_multi_step_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_multi_step_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockMultiStepLLMCommandGenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
        "llm_type": "azure",
        "llm_temperature": "0.3",
        "request_timeout": "15",
        "llm_engine": "azure-test",
        "embeddings": '{"deployment": "test"}',
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_multi_step_llm_command_generator_non_default_llm_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockMultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    model = "command"
    config = {
        "prompt": TEST_PROMPT_DIRECTORY,
        "llm": {
            "type": "cohere",
            "model": model,
            "request_timeout": 10,
            "temperature": 0.7,
        },
        "embeddings": {"model": "text-embedding-ada-002"},
    }

    mock_multi_step_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_multi_step_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockMultiStepLLMCommandGenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
        "llm_type": "cohere",
        "llm_temperature": "0.7",
        "request_timeout": "10",
        "embeddings": '{"model": "text-embedding-ada-002"}',
    }
    assert captured_span.attributes == expected_attributes


def test_tracing_multi_step_llm_command_generator_parse_commands(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    mock_multi_step_llm_command_generator = component_class(
        config={},
        model_storage=default_model_storage,
        resource=None,
    )
    mock_multi_step_llm_command_generator.parse_commands(
        actions="ChitChat() \n ChangeFlow() ",
        tracker=DialogueStateTracker(sender_id="test", slots=[]),
        flows=FlowsList(underlying_flows=[]),
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MultiStepLLMCommandGenerator.parse_commands"

    expected_attributes = {
        "commands": '[{"command": "chitchat"}, {"command": "change_flow"}]'
    }
    assert captured_span.attributes == expected_attributes


def test_tracing_multi_step_llm_command_generator_parse_commands_set_slot(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    mock_multi_step_llm_command_generator = component_class(
        config={},
        model_storage=default_model_storage,
        resource=None,
    )
    mock_multi_step_llm_command_generator.parse_commands(
        actions="SetSlot(book_restaurant_time, 7pm)",
        tracker=DialogueStateTracker(sender_id="test", slots=[]),
        flows=FlowsList(underlying_flows=[]),
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans

    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MultiStepLLMCommandGenerator.parse_commands"

    expected_attributes = {
        "commands": (
            '[{"name": "book_restaurant_time", "extractor": "LLM",'
            ' "command": "set slot", "is_slot_value_missing_or_none": false}]'
        )
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_multi_step_llm_command_generator_prompt_tokens(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockMultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    mock_multi_step_llm_command_generator = component_class(
        config={"trace_prompt_tokens": True},
        model_storage=default_model_storage,
        resource=Resource("multi-step-llm-command-generator"),
    )
    await mock_multi_step_llm_command_generator.invoke_llm("This is a test prompt.")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockMultiStepLLMCommandGenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": "gpt-4",
        "llm_type": "openai",
        "llm_temperature": "0.0",
        "request_timeout": "7",
        "embeddings": "{}",
        "len_prompt_tokens": "6",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_multi_step_llm_command_generator_prompt_tokens_non_openai(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    caplog: LogCaptureFixture,
) -> None:
    component_class = MockMultiStepLLMCommandGenerator

    instrumentation.instrument(
        tracer_provider,
        multi_step_llm_command_generator_class=component_class,
    )

    mock_multi_step_llm_command_generator = component_class(
        config={
            "trace_prompt_tokens": True,
            "llm": {"type": "cohere", "model": "command"},
        },
        model_storage=default_model_storage,
        resource=Resource("multi-step-llm-command-generator"),
    )

    with caplog.at_level(logging.WARNING):
        await mock_multi_step_llm_command_generator.invoke_llm("This is a test prompt.")
        assert (
            "Tracing prompt tokens is only supported for OpenAI models. Skipping."
            in caplog.text
        )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockMultiStepLLMCommandGenerator.invoke_llm"

    assert captured_span.attributes["len_prompt_tokens"] == "None"
