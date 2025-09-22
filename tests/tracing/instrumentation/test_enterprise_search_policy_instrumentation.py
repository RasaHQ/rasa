import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, Sequence
from unittest.mock import AsyncMock, Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.core.policies.enterprise_search_policy import (
    DEFAULT_EMBEDDINGS_CONFIG,
    EnterpriseSearchPolicy,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import LLM_API_HEALTH_CHECK_ENV_VAR
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import (
    MockInformationRetrieval,
    TestSpanExporter,
    get_model_groups,
)


@pytest.fixture
def mock_create_plain_embedder() -> Generator[Mock, None, None]:
    with patch(
        "rasa.core.policies.enterprise_search_policy.EnterpriseSearchPolicy"
        "._create_plain_embedder"
    ) as mock_function:
        yield mock_function


@pytest.fixture
def mock_faiss_store() -> Generator[Mock, None, None]:
    with patch(
        "rasa.core.policies.enterprise_search_policy.FAISS_Store",
    ) as mock_function:
        yield mock_function


@pytest.fixture
def mock_create_from_endpoint_config() -> Generator[Mock, None, None]:
    with patch(
        "rasa.core.policies.enterprise_search_policy.create_from_endpoint_config",
    ) as mock_function:
        mock_function.return_value = MockInformationRetrieval()
        yield mock_function


@patch("rasa.core.policies.enterprise_search_policy.llm_factory")
async def test_tracing_enterprise_search_policy_invoke_llm_default_config(
    mock_llm_factory: Mock,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    llm_response_object: LLMResponse,
) -> None:
    component_class = EnterpriseSearchPolicy
    vector_store = MockInformationRetrieval()

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    policy = component_class(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    mock_llm_client = Mock()
    mock_llm_client.acompletion = AsyncMock(return_value=llm_response_object)
    mock_llm_factory.return_value = mock_llm_client

    await policy._invoke_llm("")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "EnterpriseSearchPolicy._invoke_llm"

    assert captured_span.attributes == {
        "class_name": "EnterpriseSearchPolicy",
        # llm attributes
        "llm_type": "openai",
        "llm_model": "gpt-4.1-mini-2025-04-14",
        "llm_model_group_id": "None",
        "llm_temperature": "0.0",
        "llm_request_timeout": "10",
        # embeddings attributes
        "embeddings_model": "text-embedding-3-large",
        "embeddings_type": "openai",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "10",
        "embeddings": json.dumps(DEFAULT_EMBEDDINGS_CONFIG, sort_keys=True),
    }


@pytest.mark.usefixtures("mock_configuration")
@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "llm": {
                    "model": "gpt-4",
                    "request_timeout": 15,
                    "temperature": 0.7,
                },
                "embeddings": {"model": "text-embedding-3-large"},
            },
            {
                # "class_name": "EnterpriseSearchPolicy",
                # llm attributes
                "llm_type": "openai",
                "llm_model": "gpt-4",
                "llm_model_group_id": "None",
                "llm_temperature": "0.7",
                "llm_request_timeout": "15",
                # embeddings attributes
                "embeddings_model": "text-embedding-3-large",
                "embeddings_type": "openai",
                "embeddings_model_group_id": "None",
                # deprecated
                "request_timeout": "15",
                "embeddings": json.dumps(
                    {
                        "model": "text-embedding-3-large",
                        "provider": "openai",
                        # all of this is automatically filled by
                        # configuration parser
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
                "llm": {
                    "model_group": "llm-model-group",
                },
                "embeddings": {"model_group": "embedding-model-group"},
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
                "embeddings": json.dumps(get_model_groups()[1], sort_keys=True),
            },
        ),
    ],
)
@patch("rasa.core.policies.enterprise_search_policy.llm_factory")
async def test_tracing_enterprise_search_policy_invoke_llm_custom_config(
    mock_llm_factory: Mock,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    config: Dict[str, Any],
    expected: Dict[str, Any],
    monkeypatch: MonkeyPatch,
    llm_response_object: LLMResponse,
) -> None:
    """Test that the instrumentation traces custom configuration for the EnterpriseSearchPolicy."""  # noqa: E501
    # In order to avoid race conditions when tests are run on the same
    # Windows GitHub runner using multiple workers
    # (usually for different Python versions), we need to create a
    # unique temporary directory for the cache
    # and set the environment variable to point to it.
    with tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__()) as temp_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", temp_dir)
        component_class = EnterpriseSearchPolicy
        vector_store = MockInformationRetrieval()

        instrumentation.instrument(
            tracer_provider,
            policy_subclasses=[component_class],
        )

        policy = component_class(
            config=config,
            model_storage=default_model_storage,
            resource=Resource("enterprisesearchpolicy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )
        mock_llm_client = Mock()
        mock_llm_client.acompletion = AsyncMock(return_value=llm_response_object)
        mock_llm_factory.return_value = mock_llm_client

        await policy._invoke_llm("")
        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._invoke_llm"

        expected_attributes = {
            "class_name": component_class.__name__,
        }
        expected_attributes.update(expected)
        assert captured_span.attributes == expected_attributes


@patch("rasa.core.policies.enterprise_search_policy.llm_factory")
async def test_tracing_enterprise_search_policy_invoke_llm_len_prompt_tokens(
    mock_llm_factory: Mock,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
    llm_response_object: LLMResponse,
) -> None:
    """Test that the instrumentation traces ES prompt tokens for OpenAI models."""
    # In order to avoid race conditions when tests are run on the same
    # Windows GitHub runner using multiple workers
    # (usually for different Python versions), we need to create a
    # unique temporary directory for the cache
    # and set the environment variable to point to it.
    with tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__()) as temp_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", temp_dir)
        component_class = EnterpriseSearchPolicy
        vector_store = MockInformationRetrieval()

        instrumentation.instrument(
            tracer_provider,
            policy_subclasses=[component_class],
        )

        policy = component_class(
            config={"trace_prompt_tokens": True},
            model_storage=default_model_storage,
            resource=Resource("enterprisesearchpolicy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )
        mock_llm_client = Mock()
        mock_llm_client.acompletion = AsyncMock(return_value=llm_response_object)
        mock_llm_factory.return_value = Mock()
        await policy._invoke_llm("This is a test prompt.")

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._invoke_llm"

        assert captured_span.attributes == {
            "class_name": "EnterpriseSearchPolicy",
            "len_prompt_tokens": "6",
            # llm attributes
            "llm_type": "openai",
            "llm_model": "gpt-4.1-mini-2025-04-14",
            "llm_model_group_id": "None",
            "llm_temperature": "0.0",
            "llm_request_timeout": "10",
            # embeddings attributes
            "embeddings_model": "text-embedding-3-large",
            "embeddings_type": "openai",
            "embeddings_model_group_id": "None",
            # deprecated
            "request_timeout": "10",
            "embeddings": json.dumps(DEFAULT_EMBEDDINGS_CONFIG, sort_keys=True),
        }


@patch("rasa.core.policies.enterprise_search_policy.llm_factory")
async def test_tracing_enterprise_search_policy_invoke_llm_len_prompt_tokens_non_openai(
    mock_llm_factory: Mock,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
    llm_response_object: LLMResponse,
) -> None:
    """Test that the instrumentation does not trace ES prompt tokens for non-OpenAI models."""  # noqa: E501
    # In order to avoid race conditions when tests are run on the same
    # Windows GitHub runner using multiple workers
    # (usually for different Python versions), we need to create a
    # unique temporary directory for the cache
    # and set the environment variable to point to it.
    with tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__()) as temp_dir:
        monkeypatch.setenv("TIKTOKEN_CACHE_DIR", temp_dir)
        component_class = EnterpriseSearchPolicy
        vector_store = MockInformationRetrieval()

        instrumentation.instrument(
            tracer_provider,
            policy_subclasses=[component_class],
        )

        policy = component_class(
            config={
                "trace_prompt_tokens": True,
                "llm": {"provider": "cohere", "model": "command"},
            },
            model_storage=default_model_storage,
            resource=Resource("enterprisesearchpolicy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )

        with caplog.at_level(logging.WARNING):
            mock_llm_client = Mock()
            mock_llm_client.acompletion = AsyncMock(return_value=llm_response_object)
            mock_llm_factory.return_value = mock_llm_client
            await policy._invoke_llm("This is a test prompt.")
            assert (
                "Tracing prompt tokens is only supported for OpenAI models. Skipping."
                in caplog.text
            )

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._invoke_llm"

        assert captured_span.attributes["len_prompt_tokens"] == "None"


async def test_tracing_enterprise_search_policy_training_health_check(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
    mock_create_plain_embedder: Mock,
    mock_faiss_store: Mock,
    mock_perform_llm_health_check: Mock,
    mock_perform_embeddings_health_check: Mock,
) -> None:
    # In order to avoid race conditions when tests are run on the same
    # Windows GitHub runner using multiple workers
    # (usually for different Python versions), we need to create a
    # unique temporary directory for the cache
    # and set the environment variable to point to it.
    with tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__()) as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()

        example_doc = docs_dir / "example.txt"
        example_doc.write_text("This is an example document.")

        monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, "true")
        component_class = EnterpriseSearchPolicy
        vector_store = MockInformationRetrieval()

        instrumentation.instrument(
            tracer_provider,
            policy_subclasses=[component_class],
        )
        test_span_exported = TestSpanExporter(span_exporter)
        previous_num_captured_spans = (
            test_span_exported.get_previous_num_captured_spans()
        )

        policy = component_class(
            config={
                "trace_prompt_tokens": True,
                "llm": {"provider": "cohere", "model": "command"},
                "vector_store": {
                    "type": "faiss",
                    "source": str(docs_dir),
                },
            },
            model_storage=default_model_storage,
            resource=Resource("enterprisesearchpolicy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )

        policy.train(Mock(), Mock(), Mock(), Mock(), Mock())

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 2

        captured_spans = captured_spans[-2:]

        expected_attributes = {
            "api_health_check_enabled": True,
            "health_check_trigger_component": "EnterpriseSearchPolicy",
            "health_check_trigger_method": "enterprise_search_policy.train",
        }
        span_training_llm_health_check = next(
            span
            for span in captured_spans
            if span.name == "EnterpriseSearchPolicy.perform_llm_health_check"
        )
        span_training_embeddings_health_check = next(
            span
            for span in captured_spans
            if span.name == "EnterpriseSearchPolicy.perform_embeddings_health_check"
        )

        assert span_training_llm_health_check is not None
        assert span_training_embeddings_health_check is not None

        for key, value in expected_attributes.items():
            assert span_training_llm_health_check.attributes[key] == value
            assert span_training_embeddings_health_check.attributes[key] == value


@patch(
    "rasa.core.policies.enterprise_search_policy.create_from_endpoint_config",
    Mock(return_value=MockInformationRetrieval()),
)
async def test_tracing_enterprise_search_policy_inference_health_check(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
    mock_create_plain_embedder: Mock,
    mock_faiss_store: Mock,
    mock_create_from_endpoint_config: Mock,
    mock_perform_llm_health_check: Mock,
    mock_perform_embeddings_health_check: Mock,
) -> None:
    # In order to avoid race conditions when tests are run on the same
    # Windows GitHub runner using multiple workers
    # (usually for different Python versions), we need to create a
    # unique temporary directory for the cache
    # and set the environment variable to point to it.
    with tempfile.TemporaryDirectory(suffix=uuid.uuid4().__str__()) as temp_dir:
        monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, "true")
        resource_dir = os.path.join(temp_dir, "enterprisesearchpolicy")
        os.mkdir(resource_dir)

        component_class = EnterpriseSearchPolicy

        instrumentation.instrument(
            tracer_provider,
            policy_subclasses=[component_class],
        )
        test_span_exported = TestSpanExporter(span_exporter)
        previous_num_captured_spans = (
            test_span_exported.get_previous_num_captured_spans()
        )
        component_class.load(
            config={
                "trace_prompt_tokens": True,
                "llm": {"provider": "cohere", "model": "command"},
            },
            model_storage=default_model_storage,
            resource=Resource(resource_dir),
            execution_context=default_execution_context,
        )

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 2

        captured_spans = captured_spans[-2:]

        expected_attributes = {
            "api_health_check_enabled": True,
            "health_check_trigger_component": "EnterpriseSearchPolicy",
            "health_check_trigger_method": "enterprise_search_policy.load",
        }
        span_training_llm_health_check = next(
            span
            for span in captured_spans
            if span.name == "EnterpriseSearchPolicy.perform_llm_health_check"
        )
        span_training_embeddings_health_check = next(
            span
            for span in captured_spans
            if span.name == "EnterpriseSearchPolicy.perform_embeddings_health_check"
        )

        assert span_training_llm_health_check is not None
        assert span_training_embeddings_health_check is not None

        for key, value in expected_attributes.items():
            assert span_training_llm_health_check.attributes[key] == value
            assert span_training_embeddings_health_check.attributes[key] == value
