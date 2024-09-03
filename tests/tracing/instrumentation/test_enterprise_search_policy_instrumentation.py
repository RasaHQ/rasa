import uuid

import tempfile

import logging
from typing import Sequence
from unittest.mock import Mock

from pytest import LogCaptureFixture, MonkeyPatch
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.core import EnterpriseSearchPolicy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.tracing.instrumentation import instrumentation
from tests.tracing.instrumentation.conftest import MockInformationRetrieval


async def test_tracing_enterprise_search_policy_generate_llm_answer_default_config(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
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
    await policy._generate_llm_answer(llm=Mock(), prompt="")

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "EnterpriseSearchPolicy._generate_llm_answer"

    assert captured_span.attributes == {
        "class_name": "EnterpriseSearchPolicy",
        "llm_model": "gpt-3.5-turbo",
        "llm_type": "openai",
        "embeddings": "{}",
        "llm_temperature": "0.0",
        "request_timeout": "10",
    }


async def test_tracing_enterprise_search_policy_generate_llm_answer_custom_config(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
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
            config={
                "llm": {
                    "model": "gpt-4",
                    "request_timeout": 15,
                    "temperature": 0.7,
                },
                "embeddings": {"model": "text-embedding-ada-002"},
            },
            model_storage=default_model_storage,
            resource=Resource("enterprisesearchpolicy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )
        await policy._generate_llm_answer(llm=Mock(), prompt="")
        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._generate_llm_answer"

        assert captured_span.attributes == {
            "class_name": "EnterpriseSearchPolicy",
            "llm_model": "gpt-4",
            "llm_type": "openai",
            "embeddings": '{"model": "text-embedding-ada-002"}',
            "llm_temperature": "0.7",
            "request_timeout": "15",
        }


async def test_tracing_enterprise_search_policy_generate_llm_answer_len_prompt_tokens(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: MonkeyPatch,
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
        await policy._generate_llm_answer(llm=Mock(), prompt="This is a test prompt.")

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._generate_llm_answer"

        assert captured_span.attributes == {
            "class_name": "EnterpriseSearchPolicy",
            "llm_model": "gpt-3.5-turbo",
            "llm_type": "openai",
            "llm_temperature": "0.0",
            "request_timeout": "10",
            "embeddings": "{}",
            "len_prompt_tokens": "6",
        }


async def test_tracing_enterprise_search_policy_generate_llm_answer_len_prompt_tokens_non_openai(  # noqa: E501
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
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
            await policy._generate_llm_answer(
                llm=Mock(), prompt="This is a test prompt."
            )
            assert (
                "Tracing prompt tokens is only supported for OpenAI models. Skipping."
                in caplog.text
            )

        captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

        num_captured_spans = len(captured_spans) - previous_num_captured_spans
        assert num_captured_spans == 1

        captured_span = captured_spans[-1]
        assert captured_span.name == "EnterpriseSearchPolicy._generate_llm_answer"

        assert captured_span.attributes["len_prompt_tokens"] == "None"
