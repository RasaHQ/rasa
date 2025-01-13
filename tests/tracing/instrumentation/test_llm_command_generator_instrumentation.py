import json
import logging
from typing import Any, Dict, Sequence
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest import LogCaptureFixture

from rasa.dialogue_understanding.commands import (
    SetSlotCommand,
    StartFlowCommand,
)
from rasa.dialogue_understanding.generator.flow_retrieval import FlowRetrieval
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import LLM_API_HEALTH_CHECK_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import TextSlot
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.conftest import TRACING_TESTS_FIXTURES_DIRECTORY
from tests.tracing.instrumentation.conftest import (
    MockAvailableEndpoints,
    MockLLMCommandgenerator,
)
from tests.utilities import flows_from_str

TEST_PROMPT_DIRECTORY = str(TRACING_TESTS_FIXTURES_DIRECTORY / "test_prompt.jinja2")


@pytest.fixture
def mock_llm_factory(fake_llm_client: LLMClient) -> Mock:
    with patch("rasa.shared.utils.llm.llm_factory") as mock_function:
        mock_function.return_value = fake_llm_client
        yield mock_function


@pytest.fixture
def mock_embedder_factory(fake_embedding_client: EmbeddingClient) -> Mock:
    with patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.embedder_factory",
        Mock(return_value=fake_embedding_client),
    ) as mock_function:
        mock_function.return_value = fake_embedding_client
        yield mock_function


@pytest.mark.parametrize(
    "config, expected",
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
            {
                "llm_model": "gpt-4",
            },
        ),
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "model": "gpt-3.5-turbo",
                },
            },
            {
                "llm_model": "gpt-3.5-turbo",
            },
        ),
        ({"prompt": TEST_PROMPT_DIRECTORY}, {"llm_model": "gpt-4"}),
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "request_timeout": 7,
                    "temperature": 0.0,
                },
            },
            {
                "llm_model": "gpt-4",
            },
        ),
        (
            {
                "llm": {
                    "model_group": "llm-model-group",
                },
                "flow_retrieval": {
                    "embeddings": {"model_group": "embedding-model-group"},
                },
            },
            {
                # llm attributes
                "llm_type": "None",
                "llm_model": "None",
                "llm_model_group_id": "llm-model-group",
                "llm_temperature": "None",
                "llm_request_timeout": "None",
                # embeddings attributes
                "embeddings_model": "None",
                "embeddings_type": "None",
                "embeddings_model_group_id": "embedding-model-group",
                # deprecated
                "request_timeout": "None",
                "embeddings": json.dumps(
                    MockAvailableEndpoints().model_groups[1], sort_keys=True
                ),
            },
        ),
    ],
)
async def test_tracing_llm_command_generator_default_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    config: Dict[str, Any],
    expected: Dict[str, Any],
    mock_available_endpoints: MockAvailableEndpoints,
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
        # llm attributes
        "llm_type": "openai",
        "llm_model_group_id": "None",
        "llm_temperature": "0.0",
        "llm_request_timeout": "7",
        # embeddings attributes
        "embeddings_model": "text-embedding-ada-002",
        "embeddings_type": "openai",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "7",
        "embeddings": json.dumps(
            {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "api_base": None,
                "api_version": None,
                "api_type": "openai",
            },
            sort_keys=True,
        ),
    }
    expected_attributes.update(expected)
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
        "flow_retrieval": {"embeddings": {"deployment": "test"}},
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
        # llm attributes
        "llm_type": "azure",
        "llm_engine": "azure-test",
        "llm_model": model,
        "llm_model_group_id": "None",
        "llm_temperature": "0.3",
        "llm_request_timeout": "15",
        # embeddings attributes
        "embeddings_model": "None",
        "embeddings_type": "azure",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "15",
        "embeddings": json.dumps(
            {
                "provider": "azure",
                "deployment": "test",
                "model": None,
                "api_base": None,
                "api_version": None,
                "api_type": "azure",
            },
            sort_keys=True,
        ),
    }
    assert captured_span.attributes == expected_attributes


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "provider": "cohere",
                    "model": "command",
                    "request_timeout": 10,
                    "temperature": 0.7,
                },
                "flow_retrieval": {
                    "embeddings": {"model": "text-embedding-ada-002"},
                },
            },
            {
                # llm attributes
                "llm_type": "cohere",
                "llm_model": "command",
                "llm_model_group_id": "None",
                "llm_temperature": "0.7",
                "llm_request_timeout": "10",
                # embeddings attributes
                "embeddings_model": "text-embedding-ada-002",
                "embeddings_type": "openai",
                "embeddings_model_group_id": "None",
                # deprecated
                "request_timeout": "10",
                "embeddings": json.dumps(
                    {
                        "provider": "openai",
                        "model": "text-embedding-ada-002",
                        "api_base": None,
                        "api_version": None,
                        "api_type": "openai",
                    },
                    sort_keys=True,
                ),
            },
        ),
        (
            {
                "prompt": TEST_PROMPT_DIRECTORY,
                "llm": {
                    "model_group": "llm-model-group",
                },
                "flow_retrieval": {
                    "embeddings": {"model_group": "embedding-model-group"},
                },
            },
            {
                # llm attributes
                "llm_type": "None",
                "llm_model": "None",
                "llm_model_group_id": "llm-model-group",
                "llm_temperature": "None",
                "llm_request_timeout": "None",
                # embeddings attributes
                "embeddings_model": "None",
                "embeddings_type": "None",
                "embeddings_model_group_id": "embedding-model-group",
                # deprecated
                "request_timeout": "None",
                "embeddings": json.dumps(
                    MockAvailableEndpoints().model_groups[1], sort_keys=True
                ),
            },
        ),
    ],
)
async def test_tracing_llm_command_generator_non_default_llm_attrs(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    config: Dict[str, Any],
    expected: Dict[str, Any],
    mock_available_endpoints: MockAvailableEndpoints,
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
    }
    expected_attributes.update(expected)
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
        "len_prompt_tokens": "6",
        # llm attributes
        "llm_type": "openai",
        "llm_model": "gpt-4",
        "llm_model_group_id": "None",
        "llm_temperature": "0.0",
        "llm_request_timeout": "7",
        # embeddings attributes
        "embeddings_model": "text-embedding-ada-002",
        "embeddings_type": "openai",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "7",
        "embeddings": json.dumps(
            {
                "provider": "openai",
                "model": "text-embedding-ada-002",
                "api_base": None,
                "api_version": None,
                "api_type": "openai",
            },
            sort_keys=True,
        ),
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


@pytest.mark.parametrize(
    "llm_api_health_check_env_var_value",
    ["true", "false"],
)
async def test_tracing_llm_command_generator_training_health_checks(
    default_model_storage: ModelStorage,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    mock_llm_factory: Mock,
    mock_embedder_factory: Mock,
    mock_perform_llm_health_check: Mock,
    mock_perform_embeddings_health_check: Mock,
    llm_api_health_check_env_var_value: str,
    monkeypatch,
) -> None:
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, llm_api_health_check_env_var_value)
    monkeypatch.setenv(
        OPENAI_API_KEY_ENV_VAR, "api-key-in-test-llm-command-generator-instrumentation"
    )

    flows = flows_from_str(
        """
        flows:
          test_flow_a:
            description: This is a test flow.
            steps:
              - id: collect_foo
                collect: foo
                next: collect_bar
              - id: collect_bar
                collect: bar
          test_flow_b:
            description: This is a test flow.
            steps:
              - id: collect_fizz
                collect: fizz
                next: collect_buzz
              - id: collect_buzz
                collect: buzz
        """
    )
    domain = Mock()
    domain.slots = [
        TextSlot(
            name=slot_name,
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        for slot_name in ["foo", "bar", "fizz", "buzz"]
    ]

    config = {
        "llm": {"provider": "openai", "model": "test-gpt"},
        "flow_retrieval": {
            "embeddings": {"provider": "openai", "model": "test-embeddings"},
        },
    }
    component_class = MockLLMCommandgenerator
    flow_retrieval_class = FlowRetrieval
    instrumentation.instrument(
        tracer_provider,
        llm_command_generator_class=component_class,
        flow_retrieval_class=flow_retrieval_class,
    )

    mock_llm_command_generator = component_class(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("llm-command-generator"),
    )
    mock_llm_command_generator.train(
        training_data=Mock(),
        flows=flows,
        domain=domain,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore
    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    # One for LLM API health check, and other for Embeddings API health check
    assert num_captured_spans == 2
    captured_spans = captured_spans[-2:]

    expected_command_generator_train_attributes = {
        "api_health_check_enabled": llm_api_health_check_env_var_value == "true",
        "health_check_trigger_component": "LLMBasedCommandGenerator",
        "health_check_trigger_method": "llm_based_command_generator.train",
    }
    expected_flow_retrieval_train_attributes = {
        "api_health_check_enabled": llm_api_health_check_env_var_value == "true",
        "health_check_trigger_component": "FlowRetrieval",
        "health_check_trigger_method": "flow_retrieval.train",
    }

    span_training_llm_health_check = next(
        span
        for span in captured_spans
        if span.name == "MockLLMCommandgenerator.perform_llm_health_check"
    )
    span_training_embeddings_health_check = next(
        span
        for span in captured_spans
        if span.name == "FlowRetrieval.perform_embeddings_health_check"
    )

    assert span_training_llm_health_check is not None
    assert span_training_embeddings_health_check is not None

    for key, value in expected_command_generator_train_attributes.items():
        assert span_training_llm_health_check.attributes[key] == value
        assert span_training_llm_health_check.attributes[key] == value

    for key, value in expected_flow_retrieval_train_attributes.items():
        assert span_training_embeddings_health_check.attributes[key] == value
        assert span_training_embeddings_health_check.attributes[key] == value
