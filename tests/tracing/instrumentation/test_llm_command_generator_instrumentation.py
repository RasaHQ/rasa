import logging
from typing import Any, Dict, Sequence, Text

import pytest
from pytest import LogCaptureFixture
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.dialogue_understanding.commands import (
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.flows import Flow, FlowsList

from rasa.tracing.instrumentation import instrumentation
from tests.tracing.conftest import TRACING_TESTS_FIXTURES_DIRECTORY
from tests.tracing.instrumentation.conftest import MockLLMCommandgenerator

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
async def test_tracing_llm_command_generator_default_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    config: Dict[str, Any],
    model: Text,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    mock_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockLLMCommandgenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
        "llm_type": "openai",
        "llm_temperature": "0.0",
        "request_timeout": "7",
        "embeddings": "{}",
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_llm_command_generator_azure_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
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

    mock_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockLLMCommandgenerator.invoke_llm"

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


async def test_tracing_llm_command_generator_non_default_llm_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    model = "command"
    config = {
        "prompt": TEST_PROMPT_DIRECTORY,
        "llm": {
            "provider": "cohere",
            "model": model,
            "request_timeout": 10,
            "temperature": 0.7,
        },
        "embeddings": {"model": "text-embedding-ada-002"},
    }

    mock_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=None,
    )
    await mock_llm_command_generator.invoke_llm("some text")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockLLMCommandgenerator.invoke_llm"

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
        "llm_type": "cohere",
        "llm_temperature": "0.7",
        "request_timeout": "10",
        "embeddings": '{"model": "text-embedding-ada-002"}',
    }
    assert captured_span.attributes == expected_attributes


def test_tracing_llm_command_generator_check_commands_against_startable_flows(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    mock_llm_command_generator = component_class(
        config={},
        model_storage=default_model_storage,
        resource=None,
    )
    commands = [
        StartFlowCommand(flow="transfer_money"),
        SetSlotCommand(name="amount", value=100),
    ]
    mock_llm_command_generator._check_commands_against_startable_flows(
        commands=commands,
        startable_flows=FlowsList(underlying_flows=[Flow(id="transfer_money")]),
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    #
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert (
        captured_span.name
        == "MockLLMCommandgenerator._check_commands_against_startable_flows"
    )

    expected_attributes = {
        "commands": '[{"flow": "transfer_money", "command": "start flow"}, '
        '{"name": "amount", "extractor": "LLM", "command": "set slot", "is_slot_value_missing_or_none": false}]',  # noqa: E501
        "startable_flow_ids": '["transfer_money"]',
    }
    assert captured_span.attributes == expected_attributes


async def test_tracing_llm_command_generator_prompt_tokens(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    mock_llm_command_generator = component_class(
        config={"trace_prompt_tokens": True},
        model_storage=default_model_storage,
        resource=Resource("llm-command-generator"),
    )
    await mock_llm_command_generator.invoke_llm("This is a test prompt.")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockLLMCommandgenerator.invoke_llm"

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


async def test_tracing_llm_command_generator_prompt_tokens_non_openai(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    caplog: LogCaptureFixture,
) -> None:
    component_class = MockLLMCommandgenerator

    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
    )

    mock_llm_command_generator = component_class(
        config={
            "trace_prompt_tokens": True,
            "llm": {"provider": "cohere", "model": "command"},
        },
        model_storage=default_model_storage,
        resource=Resource("llm-command-generator"),
    )

    with caplog.at_level(logging.WARNING):
        await mock_llm_command_generator.invoke_llm("This is a test prompt.")
        assert (
            "Tracing prompt tokens is only supported for OpenAI models. Skipping."
            in caplog.text
        )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "MockLLMCommandgenerator.invoke_llm"

    assert captured_span.attributes["len_prompt_tokens"] == "None"
