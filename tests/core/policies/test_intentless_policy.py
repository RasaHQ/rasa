import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import Mock, patch, AsyncMock

import pytest
from pytest import MonkeyPatch
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

from rasa.core.constants import UTTER_SOURCE_METADATA_KEY
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import ChitChatStackFrame
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.forms_provider import Forms
from rasa.graph_components.providers.responses_provider import Responses
from rasa.shared.constants import ROUTE_TO_CALM_SLOT, PROMPT_CONFIG_KEY
from rasa.shared.core.domain import ActionNotFoundException, Domain
from rasa.shared.core.events import ActiveLoop, BotUttered, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.importers.importer import FlowSyncImporter
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.embedding.embedding_client import EmbeddingClient
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.llm import tracker_as_readable_transcript
from rasa.core.policies.intentless_policy import (
    Conversation,
    IntentlessPolicy,
    Interaction,
    action_from_response,
    conversation_as_prompt,
    conversation_samples_from_trackers,
    filter_responses,
    truncate_documents,
)

UTTER_GREET_ACTION = "utter_greet"
GREET_INTENT_NAME = "greet"

TEST_DOMAIN = """
forms:
  test_form:
    required_slots:
    - question1
  another_form:
    required_slots:
    - q2

responses:
  utter_greet:
    - text: Hi there!
  utter_goodbye:
    - text: Bye!
  utter_chitchat/ask_weather:
    - text: It's sunny where I live
  utter_chitchat/ask_name:
    - text: I am Mr. Bot
"""

TEST_NLU = """
version: "3.1"
nlu:
  - intent: greet
    examples: |
      - hey
      - hello
      - hi
      - hello there
  - intent: goodbye
    examples: |
      - see you!
      - bye
"""


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
def intentless_policy(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[IntentlessPolicy, None, None]:
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


def test_action_from_response() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    text = "foobar"
    assert action_from_response(text, responses) == "utter_foobar"


def test_action_from_response_not_found() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    text = "foobarbaz"
    assert action_from_response(text, responses) is None


def test_action_from_response_empty_responses() -> None:
    responses: Dict[str, List[Dict[str, Any]]] = {}
    text = "foobarbaz"
    assert action_from_response(text, responses) is None


def test_action_from_response_empty_text() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    assert action_from_response("", responses) is None
    assert action_from_response(None, responses) is None


def test_conversation_samples_from_tracker() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    tracker = DialogueStateTracker.from_events(
        "test",
        [UserUttered("hello"), BotUttered("Hey there! How can I help you?")],
    )
    samples = conversation_samples_from_trackers([tracker], responses)
    assert len(samples) == 1
    assert samples[0].interactions == [
        Interaction(text="hello", actor="USER"),
        Interaction(text="Hey there! How can I help you?", actor="AI"),
    ]


def test_conversation_samples_from_no_trackers() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    samples = conversation_samples_from_trackers([], responses)
    assert len(samples) == 0


def test_conversation_samples_filters_single_turn_interactions() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hello"),
        ],
    )
    samples = conversation_samples_from_trackers([tracker], responses)
    assert len(samples) == 0


def test_conversation_samples_filters_only_bot_or_only_user_interactions() -> None:
    responses = {
        "utter_greet": [
            {
                "text": "Hey there! How can I help you?",
            }
        ],
        "utter_foobar": [
            {
                "text": "foobar",
            }
        ],
    }
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hello"),
            UserUttered("hello"),
        ],
    )
    tracker2 = DialogueStateTracker.from_events(
        "test2",
        [
            BotUttered("Hey there! How can I help you?"),
            BotUttered("Hey there! How can I help you?"),
        ],
    )
    samples = conversation_samples_from_trackers([tracker, tracker2], responses)
    assert len(samples) == 0


def test_truncate_documents() -> None:
    docs = [
        Document(page_content="hello world how are you"),
        Document(page_content="foo bar baz"),
        Document(page_content="foobaz barbaz foobar"),
    ]
    r = truncate_documents(docs, max_number_of_tokens=10)
    assert len(r) == 2


def test_truncate_documents_no_truncation() -> None:
    docs = [
        Document(page_content="hello world how are you"),
        Document(page_content="foo bar baz"),
        Document(page_content="foobaz barbaz foobar"),
    ]
    r = truncate_documents(docs, max_number_of_tokens=100)
    assert len(r) == 3


def test_truncate_documents_no_documents() -> None:
    docs: List[Document] = []
    r = truncate_documents(docs, max_number_of_tokens=100)
    assert len(r) == 0


def test_conversation_as_prompt() -> None:
    conversation = Conversation(
        interactions=[
            Interaction(text="hello", actor="USER"),
            Interaction(text="Hey there! How can I help you?", actor="AI"),
        ]
    )
    prompt = conversation_as_prompt(conversation)
    assert prompt == "USER: hello\nAI: Hey there! How can I help you?"


def test_conversation_as_prompt_with_empty_conversation() -> None:
    conversation = Conversation(interactions=[])
    prompt = conversation_as_prompt(conversation)
    assert prompt == ""


def test_train_intentless_policy(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)
    resource = intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    loaded = IntentlessPolicy.load(
        IntentlessPolicy.get_default_config(),
        default_model_storage,
        resource,
        default_execution_context,
    )

    assert loaded is not None
    assert loaded.response_index is not None
    assert loaded.conversation_samples_index is not None


async def test_intentless_policy_predicts(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hello"),
        ],
    )
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    policy_prediction = await intentless_policy.predict_action_probabilities(
        tracker, domain
    )

    assert policy_prediction.policy_name == "IntentlessPolicy"
    assert any(p != 0.0 for p in policy_prediction.probabilities)
    # doesn't hold true since the fake llms embeddings are not normalized
    # assert all(p >= 0.0 and p <=1.0 for p in policy_prediction.probabilities)
    assert policy_prediction.action_metadata == {
        UTTER_SOURCE_METADATA_KEY: intentless_policy.__class__.__name__
    }


async def test_intentless_policy_predicts_loop(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hello"),
            BotUttered("Hi there!"),
            ActiveLoop("test_form"),
        ],
    )
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    policy_prediction = await intentless_policy.predict_action_probabilities(
        tracker, domain
    )

    try:
        action_domain_index = domain.index_for_action("test_form")
    except ActionNotFoundException:
        assert False
    assert policy_prediction.policy_name == "IntentlessPolicy"
    assert policy_prediction.probabilities[action_domain_index] > 0.0


async def test_intentless_policy_predicts_listen_if_no_loop(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hello"),
            BotUttered("Hi there!"),
        ],
    )
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    policy_prediction = await intentless_policy.predict_action_probabilities(
        tracker, domain
    )

    try:
        action_domain_index = domain.index_for_action("action_listen")
    except ActionNotFoundException:
        assert False
    assert policy_prediction.policy_name == "IntentlessPolicy"
    assert policy_prediction.probabilities[action_domain_index] > 0.0


def test_select_few_shot_conversations(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hi"),
        ],
    )
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    max_number_of_tokens = 100
    number_of_samples = 2
    history = tracker_as_readable_transcript(tracker)
    conversations = intentless_policy.select_few_shot_conversations(
        history, number_of_samples, max_number_of_tokens
    )
    assert len(conversations) == number_of_samples
    # can be either one or two as duplicates are removed so of both are the same
    # only one is returned
    assert len(intentless_policy.extract_ai_responses(conversations)) <= 2


def test_select_few_shot_conversations_with_empty_history(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    max_number_of_tokens = 100
    number_of_samples = 2
    history = ""
    conversations = intentless_policy.select_few_shot_conversations(
        history, number_of_samples, max_number_of_tokens
    )
    assert len(conversations) == 0
    assert len(intentless_policy.extract_ai_responses(conversations)) == 0


def test_select_few_shot_conversations_with_empty_samples(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hi"),
        ],
    )
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)

    intentless_policy.train([], domain, responses, forms, TrainingData(), FlowsList([]))

    max_number_of_tokens = 100
    number_of_samples = 0
    history = tracker_as_readable_transcript(tracker)
    conversations = intentless_policy.select_few_shot_conversations(
        history, number_of_samples, max_number_of_tokens
    )
    assert len(conversations) == 0
    assert len(intentless_policy.extract_ai_responses(conversations)) == 0


def test_select_response_examples_with_empty_responses(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data={})
    forms = Forms(data=domain.forms)
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hi"),
        ],
    )

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    max_number_of_tokens = 100
    number_of_samples = 2
    history = tracker_as_readable_transcript(tracker)
    example_responses = intentless_policy.select_response_examples(
        history, number_of_samples, max_number_of_tokens
    )
    assert len(example_responses) == 0


def test_select_response_examples(
    intentless_policy: IntentlessPolicy,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    domain = Domain.from_yaml(TEST_DOMAIN)
    responses = Responses(data=domain.responses)
    forms = Forms(data=domain.forms)
    tracker = DialogueStateTracker.from_events(
        "test",
        [
            UserUttered("hi"),
        ],
    )

    intentless_policy.train(
        trackers_for_training(), domain, responses, forms, TrainingData(), FlowsList([])
    )

    max_number_of_tokens = 100
    number_of_samples = 2
    history = tracker_as_readable_transcript(tracker)
    example_responses = intentless_policy.select_response_examples(
        history, number_of_samples, max_number_of_tokens
    )
    assert len(example_responses) == 2


def test_response_filtering_user_flows() -> None:
    domain = Domain.load([os.path.join("data", "train", "domain.yml")])
    utter_have_stars = "utter_i_have_stars"
    utter_hi = "utter_hi_first_name"
    user_flows = flows_from_str(
        f"""
        flows:
          foo:
            description: test foo.
            steps:
              - action: {utter_hi}
          bar:
            description: test bar.
            steps:
              - action: {utter_have_stars}
        """
    )
    filtered_responses = filter_responses(
        Responses(domain.responses), Forms(domain.forms), user_flows
    )
    assigned_responses = {
        utter_hi,
        utter_have_stars,
        "utter_ask_age",
        "utter_ask_first_name",
    }
    unassigned_responses = {
        response for response in domain.responses if response not in assigned_responses
    }
    assert set(filtered_responses.data.keys()) == unassigned_responses


def test_response_filtering_default_flows() -> None:
    default_flows = FlowSyncImporter.load_default_pattern_flows()
    domain = FlowSyncImporter.load_default_pattern_flows_domain()

    filtered_responses = filter_responses(
        Responses(domain.responses), Forms(domain.forms), default_flows
    )
    assert len(domain.responses) > 0
    assert len(default_flows.utterances) > 0

    num_unused_default_utterances = len(domain.responses) - len(
        default_flows.utterances
    )

    assert len(filtered_responses.data.keys()) == num_unused_default_utterances


# indirect parametrization of the fixture
async def test_intentless_policy_prompt_init_custom(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            config = {
                **IntentlessPolicy.get_default_config(),
                PROMPT_CONFIG_KEY: "data/prompt_templates/test_prompt.jinja2",
            }
            intentless_policy = IntentlessPolicy.create(
                config,
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )
            assert intentless_policy.prompt_template.startswith(
                "Identify the user's message"
            )

            domain = Domain.from_yaml(TEST_DOMAIN)
            responses = Responses(data=domain.responses)
            forms = Forms(data=domain.forms)
            resource = intentless_policy.train(
                trackers_for_training(),
                domain,
                responses,
                forms,
                TrainingData(),
                FlowsList([]),
            )
            loaded = IntentlessPolicy.load(
                IntentlessPolicy.get_default_config(),
                default_model_storage,
                resource,
                default_execution_context,
            )
            assert loaded.prompt_template.startswith("Identify the user's message")


async def test_intentless_policy_prompt_init_default(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            intentless_policy = IntentlessPolicy(
                IntentlessPolicy.get_default_config(),
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )
            assert intentless_policy.prompt_template.startswith(
                "The following is a friendly conversation"
            )

            domain = Domain.from_yaml(TEST_DOMAIN)
            responses = Responses(data=domain.responses)
            forms = Forms(data=domain.forms)
            resource = intentless_policy.train(
                trackers_for_training(),
                domain,
                responses,
                forms,
                TrainingData(),
                FlowsList([]),
            )
            loaded = IntentlessPolicy.load(
                IntentlessPolicy.get_default_config(),
                default_model_storage,
                resource,
                default_execution_context,
            )
            assert loaded.prompt_template.startswith(
                "The following is a friendly conversation"
            )


async def test_intentless_policy_fingerprint_addon_diff_in_prompt_template(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
) -> None:
    prompt_dir = Path(tmp_path) / PROMPT_CONFIG_KEY
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "intentless_policy_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {
        **IntentlessPolicy.get_default_config(),
        PROMPT_CONFIG_KEY: str(prompt_file),
    }
    print(config)
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            intentless_policy = IntentlessPolicy(
                config,
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )

    fingerprint_1 = intentless_policy.fingerprint_addon(config)

    prompt_file.write_text("This is a test prompt. It has been changed.")

    fingerprint_2 = intentless_policy.fingerprint_addon(config)
    assert fingerprint_1 != fingerprint_2


async def test_intentless_policy_fingerprint_addon_no_diff_in_prompt_template(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    tmp_path: Path,
) -> None:
    prompt_dir = Path(tmp_path) / PROMPT_CONFIG_KEY
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "intentless_policy_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {
        **IntentlessPolicy.get_default_config(),
        PROMPT_CONFIG_KEY: str(prompt_file),
    }
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            intentless_policy = IntentlessPolicy(
                config,
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )

    fingerprint_1 = intentless_policy.fingerprint_addon(config)
    fingerprint_2 = intentless_policy.fingerprint_addon(config)
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2


async def test_intentless_policy_fingerprint_addon_default_prompt_template(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    with patch(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    ):
        with patch(
            "rasa.core.policies.intentless_policy.embedder_factory",
            Mock(return_value=fake_embedding_client),
        ):
            intentless_policy = IntentlessPolicy(
                IntentlessPolicy.get_default_config(),
                default_model_storage,
                Resource("intentless_policy"),
                default_execution_context,
            )
    fingerprint_1 = intentless_policy.fingerprint_addon({})
    fingerprint_2 = intentless_policy.fingerprint_addon({})
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2


async def test_intentless_policy_abstains_in_coexistence(
    fake_llm_client: LLMClient,
    fake_embedding_client: EmbeddingClient,
    monkeypatch: MonkeyPatch,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> None:
    """Test that the policy abstains in coexistence.

    If the conversation is already handled by the nlu system, IntentlessPolicy
    does not make a prediction.
    """
    test_domain = Domain.from_yaml(
        """
        responses:
            utter_greet:
                - text: Hi there!
            utter_goodbye:
                - text: Bye!
        """
    )

    stack = DialogueStack(frames=[ChitChatStackFrame()])

    tracker = DialogueStateTracker.from_events(
        "test abstain",
        domain=test_domain,
        slots=test_domain.slots
        + [BooleanSlot(ROUTE_TO_CALM_SLOT, mappings=[], initial_value=False)],
        evts=[UserUttered("hello")],
    )
    tracker.update_stack(stack)

    monkeypatch.setattr(
        "rasa.core.policies.intentless_policy.llm_factory",
        Mock(return_value=fake_llm_client),
    )

    monkeypatch.setattr(
        "rasa.core.policies.intentless_policy.embedder_factory",
        Mock(return_value=fake_embedding_client),
    )

    test_policy = IntentlessPolicy.create(
        IntentlessPolicy.get_default_config(),
        default_model_storage,
        Resource("intentless_policy"),
        default_execution_context,
    )
    mock_response_index = AsyncMock(spec=FAISS)
    monkeypatch.setattr(test_policy, "response_index", mock_response_index)

    mock_conversation_samples_index = AsyncMock(spec=FAISS)
    monkeypatch.setattr(
        test_policy, "conversation_samples_index", mock_conversation_samples_index
    )

    monkeypatch.setattr(
        test_policy, "find_closest_response", AsyncMock(return_value=("Hi there!", 1.0))
    )

    prediction = await test_policy.predict_action_probabilities(
        tracker=tracker, domain=test_domain
    )

    # check that the policy didn't predict anything
    assert prediction.max_confidence == 0.0


@pytest.mark.parametrize(
    "routing_slot_value,result",
    [
        (None, True),
        (True, False),
        (False, True),
    ],
)
def test_should_abstain_in_coexistence(
    routing_slot_value: Optional[bool],
    result: bool,
    intentless_policy: IntentlessPolicy,
):
    tracker = DialogueStateTracker(
        "id1",
        slots=[BooleanSlot(ROUTE_TO_CALM_SLOT, [], initial_value=routing_slot_value)],
    )

    assert result == intentless_policy.should_abstain_in_coexistence(tracker, True)
