import json
import logging
from typing import Any, Dict, Generator, List, Optional, Sequence
from unittest.mock import Mock, patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest import LogCaptureFixture, MonkeyPatch

from rasa.core.policies.intentless_policy import IntentlessPolicy
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.forms_provider import Forms
from rasa.graph_components.providers.responses_provider import Responses
from rasa.shared.constants import LLM_API_HEALTH_CHECK_ENV_VAR, OPENAI_API_KEY_ENV_VAR
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import BotUttered, DialogueStackUpdated, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.llm import LLMInput
from rasa.tracing.instrumentation import instrumentation
from tests.core.policies.test_intentless_policy import TEST_DOMAIN


@pytest.fixture
def mock_llm_factory(fake_llm_client: LLMClient) -> Mock:
    with patch("rasa.core.policies.intentless_policy.llm_factory") as mock_function:
        mock_function.return_value = fake_llm_client
        yield mock_function


@pytest.fixture
def mock_embedder_factory(fake_embedding_client: EmbeddingClient) -> Mock:
    with patch(
        "rasa.core.policies.intentless_policy.embedder_factory",
        Mock(return_value=fake_embedding_client),
    ) as mock_function:
        mock_function.return_value = fake_embedding_client
        yield mock_function


@pytest.fixture
def trackers_for_training() -> List[TrackerWithCachedStates]:
    return [
        TrackerWithCachedStates.from_events(
            "test",
            [UserUttered("hello"), BotUttered("Hi there!")],
        ),
        TrackerWithCachedStates.from_events(
            "test2",
            [UserUttered("hi"), BotUttered("Hi there!")],
        ),
        TrackerWithCachedStates.from_events(
            "test3",
            [UserUttered("goodybe"), BotUttered("Bye!")],
        ),
    ]


@pytest.fixture
def intentless_policy_generator(
    mock_llm_factory: Mock,
    mock_embedder_factory: Mock,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[IntentlessPolicy, None, None]:
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
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
    mock_tracker = Mock()

    await intentless_policy_generator.generate_answer(
        ["Howdy!"], [""], "", mock_tracker
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
    monkeypatch.setattr(
        "rasa.tracing.instrumentation.attribute_extractors.resolve_tiktoken_encode",
        lambda model_name, fallback_encoding="cl100k_base": (
            lambda prompt: [1, 2, 3, 4]
        ),
    )
    component_class = IntentlessPolicy

    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    intentless_policy_generator.trace_prompt_tokens = True

    await intentless_policy_generator._generate_llm_answer(
        Mock(), LLMInput(prompt="This is a test prompt.", metadata={})
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore

    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 1

    captured_span = captured_spans[-1]

    assert captured_span.name == "IntentlessPolicy._generate_llm_answer"

    assert captured_span.attributes == {
        "class_name": "IntentlessPolicy",
        "len_prompt_tokens": "4",
        # llm attributes
        "llm_type": "openai",
        "llm_model": "gpt-5-mini-2025-08-07",
        "llm_model_group_id": "None",
        "llm_temperature": "1.0",
        "llm_request_timeout": "5",
        # embeddings attributes
        "embeddings_model": "text-embedding-3-large",
        "embeddings_type": "openai",
        "embeddings_model_group_id": "None",
        # deprecated
        "request_timeout": "5",
        "embeddings": json.dumps(
            {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "api_base": None,
                "api_version": None,
                "api_type": "openai",
            },
            sort_keys=True,
        ),
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
            Mock(), LLMInput(prompt="This is a test prompt.", metadata={})
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


@pytest.mark.parametrize(
    "llm_api_health_check_env_var_value",
    ["true", "false"],
)
async def test_tracing_intentless_policy_training_and_inference_health_check(
    intentless_policy_generator: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    previous_num_captured_spans: int,
    monkeypatch: MonkeyPatch,
    trackers_for_training: List[TrackerWithCachedStates],
    llm_api_health_check_env_var_value: str,
    mock_perform_llm_health_check: Mock,
    mock_perform_embeddings_health_check: Mock,
) -> None:
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, llm_api_health_check_env_var_value)
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    component_class = IntentlessPolicy
    instrumentation.instrument(
        tracer_provider,
        policy_subclasses=[component_class],
    )

    resource = intentless_policy_generator.train(
        trackers_for_training,
        domain,
        responses,
        forms,
        TrainingData(),
        FlowsList([]),
    )
    IntentlessPolicy.load(
        IntentlessPolicy.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )

    captured_spans: Sequence[ReadableSpan] = span_exporter.get_finished_spans()  # type: ignore
    num_captured_spans = len(captured_spans) - previous_num_captured_spans
    assert num_captured_spans == 4  # Two from training and two from inference
    captured_spans = captured_spans[-4:]

    expected_train_attributes = {
        "api_health_check_enabled": llm_api_health_check_env_var_value == "true",
        "health_check_trigger_component": "IntentlessPolicy",
    }

    span_training_llm_health_check = next(
        span
        for span in captured_spans
        if span.name == "IntentlessPolicy.perform_llm_health_check"
        and span.attributes.get("health_check_trigger_method")
        == "intentless_policy.train"
    )
    span_training_embeddings_health_check = next(
        span
        for span in captured_spans
        if span.name == "IntentlessPolicy.perform_embeddings_health_check"
        and span.attributes.get("health_check_trigger_method")
        == "intentless_policy.train"
    )
    span_inference_llm_health_check = next(
        span
        for span in captured_spans
        if span.name == "IntentlessPolicy.perform_llm_health_check"
        and span.attributes.get("health_check_trigger_method")
        == "intentless_policy.load"
    )
    span_inference_embeddings_health_check = next(
        span
        for span in captured_spans
        if span.name == "IntentlessPolicy.perform_embeddings_health_check"
        and span.attributes.get("health_check_trigger_method")
        == "intentless_policy.load"
    )

    assert span_training_llm_health_check is not None
    assert span_training_embeddings_health_check is not None
    assert span_inference_llm_health_check is not None
    assert span_inference_embeddings_health_check is not None

    for key, value in expected_train_attributes.items():
        assert span_training_llm_health_check.attributes[key] == value
        assert span_training_embeddings_health_check.attributes[key] == value
        assert span_inference_llm_health_check.attributes[key] == value
        assert span_inference_embeddings_health_check.attributes[key] == value
