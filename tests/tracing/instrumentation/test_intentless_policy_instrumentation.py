import logging
from typing import Any, Dict, Generator, Optional, Sequence
from unittest.mock import Mock, patch

import pytest
from pytest import MonkeyPatch
from pytest import LogCaptureFixture
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.tracing.instrumentation import instrumentation


@pytest.fixture
def intentless_policy_generator(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[IntentlessPolicy, None, None]:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            yield IntentlessPolicy.create(
                IntentlessPolicy.get_default_config(),
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )


async def test_tracing_intentless_policy_generate_answer(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    await intentless_policy_generator.generate_answer(
        ["Howdy!"],
        [""],
        "",
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    # includes the child span for `_generate_llm_answer` method call
    assert num_captured_spans == 2

    captured_span = captured_spans[-1]

    assert captured_span.name == "IntentlessPolicy.generate_answer"

    assert captured_span.attributes == {"llm_response": '"Hello there!"'}


def test_tracing_intentless_policy_extract_ai_responses(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    ai_reply_one = "With our service you can send money for free to friends and family."
    ai_reply_two = "At this point in time we only support domestic transfers."

    conversation_samples = [
        f"USER: Are there any fees associated with sending money?\n"
        f"AI: {ai_reply_one}",
        f"USER: Are transfers free on this app?\n" f"AI: {ai_reply_one}",
        f"USER: Do you support international transfers?\n" f"AI: {ai_reply_two}",
    ]
    intentless_policy_generator.extract_ai_responses(conversation_samples)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "IntentlessPolicy.extract_ai_responses"

    expected_attributes = {"ai_responses": f'["{ai_reply_one}", "{ai_reply_two}"]'}

    assert captured_span.attributes == expected_attributes


def test_tracing_intentless_policy_select_few_shot_conversations(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")

    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator.select_few_shot_conversations(
        "",
        2,
        100,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "IntentlessPolicy.select_few_shot_conversations"

    expected_attributes = {"conversation_samples": "[]"}

    assert captured_span.attributes == expected_attributes


def test_tracing_intentless_policy_select_response_examples(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator.select_response_examples(
        "",
        3,
        100,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "IntentlessPolicy.select_response_examples"

    expected_attributes = {"ai_response_examples": "[]"}

    assert captured_span.attributes == expected_attributes


async def test_tracing_intentless_policy_find_closest_response(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    tracker = DialogueStateTracker.from_events(
        "test_sender",
        [
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/0", "value": {"frame_id": "ZG16JCGM", "flow_id": "replace_card", "step_id": "START", "frame_type": "regular", "type": "flow"}}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/0/step_id", "value": "0_collect_confirm_correct_card"}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/1", "value": {"frame_id": "ORK38NAU", "flow_id": "pattern_collect_information", "step_id": "START", "collect": "confirm_correct_card", "utter": "utter_ask_confirm_correct_card",  "collect_action": "action_ask_confirm_correct_card", "rejections": [], "type": "pattern_collect_information"}}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/1/step_id", "value": "start"}]'
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/1/step_id", "value": "1_validate_{{context.collect}}"}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/1/step_id", "value": "ask_collect"}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/1/step_id", "value": "3_action_listen"}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/2", "value": {"frame_id": "VL82F9JD", "flow_id": "pattern_chitchat", "step_id": "START", "type": "pattern_chitchat"}}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "replace", "path": "/2/step_id", "value": "0_action_trigger_chitchat"}]'  # noqa: E501
            ),
            DialogueStackUpdated(
                update='[{"op": "add", "path": "/3", "value": {"frame_id": "F053RNE8", "type": "chitchat"}}]'  # noqa: E501
            ),
        ],
    )

    await intentless_policy_generator.find_closest_response(tracker)

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "IntentlessPolicy.find_closest_response"

    assert captured_span.attributes == {
        "current_context": '{"frame_id": "F053RNE8", "type": "chitchat"}'
    }


@pytest.mark.parametrize(
    "action_name, expected_attributes",
    [
        ("action_listen", {"action_name": "action_listen", "score": 1.0}),
        (None, {"action_name": "null", "score": 1.0}),
    ],
)
def test_tracing_intentless_policy_prediction_result(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    action_name: Optional[str],
    expected_attributes: Dict[str, Any],
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator._prediction_result(action_name, Domain.empty())

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]
    assert captured_span.name == "IntentlessPolicy._prediction_result"

    assert captured_span.attributes == expected_attributes


async def test_tracing_intentless_policy_generate_llm_answer_len_prompt_tokens(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator.trace_prompt_tokens = True

    await intentless_policy_generator._generate_llm_answer(
        Mock(), "This is a test prompt."
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "IntentlessPolicy._generate_llm_answer"

    assert captured_span.attributes == {
        "class_name": "IntentlessPolicy",
        "llm_model": "gpt-3.5-turbo",
        "llm_type": "openai",
        "embeddings": '{"provider": "openai", "model": "text-embedding-ada-002"}',
        "llm_temperature": "0.0",
        "request_timeout": "5",
        "len_prompt_tokens": "6",
    }


async def test_intentless_policy_generate_llm_answer_len_prompt_tokens_non_openai(
    intentless_policy_generator: IntentlessPolicy,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    caplog: LogCaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator.trace_prompt_tokens = True
    intentless_policy_generator.config = {
        "llm": {"provider": "cohere", "model": "command"}
    }

    with caplog.at_level(logging.WARNING):
        await intentless_policy_generator._generate_llm_answer(
            Mock(), "This is a test prompt."
        )
        assert (
            "Tracing prompt tokens is only supported for OpenAI models. Skipping."
            in caplog.text
        )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "IntentlessPolicy._generate_llm_answer"

    assert captured_span.attributes["len_prompt_tokens"] == "None"
