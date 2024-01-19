from typing import Any, Dict, Sequence, Text

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rasa.engine.storage.storage import ModelStorage

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
                "llm": {"model_name": "gpt-4", "request_timeout": 7},
            },
            "gpt-4",
        ),
        ({"prompt": TEST_PROMPT_DIRECTORY}, "None"),
    ],
)
@pytest.mark.asyncio
async def test_tracing_llm_command_generator(
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
    mock_llm_command_generator._generate_action_list_using_llm("some text")

    captured_spans: Sequence[
        ReadableSpan
    ] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert (
        captured_span.name == "MockLLMCommandgenerator._generate_action_list_using_llm"
    )

    expected_attributes = {
        "class_name": component_class.__name__,
        "llm_model": model,
    }
    assert captured_span.attributes == expected_attributes
