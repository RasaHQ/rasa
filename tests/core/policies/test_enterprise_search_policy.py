import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import structlog
from _pytest.logging import LogCaptureFixture
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms.fake import FakeListLLM
from pytest import MonkeyPatch

import rasa.shared.utils.io
from rasa.core.constants import UTTER_SOURCE_METADATA_KEY
from rasa.core.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
    SearchResult,
    SearchResultList,
)
from rasa.core.policies.enterprise_search_policy import (
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE,
    DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_RELEVANCY_CHECK_AND_CITATION_TEMPLATE,
    ENTERPRISE_SEARCH_CONFIG_FILE_NAME,
    SEARCH_QUERY_METADATA_KEY,
    SEARCH_RESULTS_METADATA_KEY,
    EnterpriseSearchPolicy,
    VectorStoreConfigurationError,
)
from rasa.core.policies.enterprise_search_policy_config import (
    CHECK_RELEVANCY_PROPERTY,
    DEFAULT_EMBEDDINGS_CONFIG,
    DEFAULT_LLM_CONFIG,
    DEFAULT_USE_LLM_PROPERTY,
    USE_LLM_PROPERTY,
)
from rasa.core.policies.policy import PolicyPrediction
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    ChitChatStackFrame,
    DialogueStackFrame,
    SearchStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.exceptions import EnterpriseSearchPolicyError
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_GROUP_CONFIG_KEY,
    OPENAI_API_KEY_ENV_VAR,
    RASA_PATTERN_CANNOT_HANDLE_NO_RELEVANT_ANSWER,
    ROUTE_TO_CALM_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, BotUttered, UserUttered
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_LLM_RESPONSE_METADATA,
    KEY_PROMPT_NAME,
    KEY_USER_PROMPT,
    PROMPTS,
)
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.llm import get_prompt_template
from tests.utilities import filter_logs


@pytest.fixture
def vector_store() -> InformationRetrieval:
    return MagicMock(spec=InformationRetrieval)


@pytest.fixture()
def resource() -> Resource:
    return Resource("enterprise_search_policy")


@pytest.fixture()
def default_enterprise_search_policy(
    resource: Resource,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
) -> EnterpriseSearchPolicy:
    return EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=resource,
        execution_context=default_execution_context,
    )


@pytest.fixture()
def enterprise_search_tracker() -> DialogueStateTracker:
    domain = Domain.empty()
    dialogue_stack = DialogueStack(
        frames=[
            SearchStackFrame(frame_id="foobar"),
        ]
    )
    # create a tracker with the stack set
    tracker = DialogueStateTracker.from_events(
        "test_policy_prediction",
        domain=domain,
        slots=domain.slots,
        evts=[UserUttered("what is the meaning of life?")],
    )
    tracker.update_stack(dialogue_stack)
    return tracker


@pytest.fixture
def mocked_enterprise_search_policy(
    monkeypatch,
    resource: Resource,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
):
    monkeypatch.setenv(
        OPENAI_API_KEY_ENV_VAR, "mock key in test_enterprise_search_policy"
    )
    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=resource,
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    return policy


@pytest.fixture
def mock_create_prediction_internal_error():
    with patch.object(
        EnterpriseSearchPolicy,
        "_create_prediction_internal_error",
        return_value=MagicMock(),
    ) as mock_create_prediction_internal_error:
        yield mock_create_prediction_internal_error


@pytest.fixture
def mock_create_prediction_cannot_handle():
    with patch.object(
        EnterpriseSearchPolicy,
        "_create_prediction_cannot_handle",
        return_value=MagicMock(),
    ) as mock_create_prediction_cannot_handle:
        yield mock_create_prediction_cannot_handle


@pytest.fixture
def mock_is_llm_response_relevant():
    with patch.object(
        EnterpriseSearchPolicy,
        "_is_llm_response_relevant",
        return_value=MagicMock(),
    ) as mock_is_llm_response_relevant:
        yield mock_is_llm_response_relevant


@pytest.fixture
def search_results() -> SearchResultList:
    return SearchResultList(
        results=[
            SearchResult(
                text="test query",
                metadata={"id": "doc1", "answer": "test response"},
            ),
            SearchResult(
                text="test query2",
                metadata={"id": "doc2", "answer": "world response"},
            ),
        ],
        metadata={},
    )


@pytest.mark.parametrize(
    "config,prompt_starts_with,prompt_contains",
    [
        # Use of deprecated 'prompt' key
        (
            {"prompt": "data/prompt_templates/test_prompt.jinja2"},
            "Identify the user's message intent",
            "",
        ),
        (
            {
                "prompt": "data/prompt_templates/test_prompt.jinja2",
                "citation_enabled": True,
            },
            "Identify the user's message intent",
            "",
        ),
        (
            {
                "prompt": "data/prompt_templates/test_prompt.jinja2",
                "check_relevancy": True,
            },
            "Identify the user's message intent",
            "",
        ),
        # Use of `prompt_template' config key
        (
            {"prompt_template": "data/prompt_templates/test_prompt.jinja2"},
            "Identify the user's message intent",
            "",
        ),
        (
            {
                "prompt_template": "data/prompt_templates/test_prompt.jinja2",
                "citation_enabled": True,
            },
            "Identify the user's message intent",
            "",
        ),
        (
            {
                "prompt_template": "data/prompt_templates/test_prompt.jinja2",
                "check_relevancy": True,
            },
            "Identify the user's message intent",
            "",
        ),
        # Use of default prompts based on the citation and relevancy check
        (
            {},
            "Given the following information, please provide an answer based on"
            " the provided documents",
            "",
        ),
        (
            {"citation_enabled": True},
            "Given the following information, please provide an answer based on"
            " the provided documents",
            "Citing Sources",
        ),
        (
            {"check_relevancy": True},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            "[NO_RAG_ANSWER]",
        ),
        (
            {"check_relevancy": True, "citation_enabled": True},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            "[NO_RAG_ANSWER]",
        ),
        (
            {"check_relevancy": True, "citation_enabled": False},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            "[NO_RAG_ANSWER]",
        ),
    ],
)
async def test_enterprise_search_policy_prompt(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    config: dict,
    prompt_starts_with: str,
    prompt_contains: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that the prompt is set correctly based on the config."""
    monkeypatch.setenv(
        OPENAI_API_KEY_ENV_VAR, "mock key in test_enterprise_search_policy"
    )
    policy = EnterpriseSearchPolicy(
        config={**config, **{"vector_store": {"type": "milvus"}}},
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.prompt_template.startswith(prompt_starts_with)
    assert prompt_contains in policy.prompt_template
    with patch(
        "rasa.core.policies.enterprise_search_policy.llm_factory",
        Mock(return_value=FakeListLLM(responses=["Hello there", "Goodbye"])),
    ):
        with patch(
            "rasa.core.policies.enterprise_search_policy.embedder_factory",
            Mock(return_value=FakeEmbeddings(size=100)),
        ):
            resource = policy.train([], Domain.empty(), None, None, None)
            loaded = EnterpriseSearchPolicy.load(
                {**config, **{"vector_store": {"type": "milvus"}}},
                default_model_storage,
                resource,
                default_execution_context,
            )
    assert loaded.prompt_template.startswith(prompt_starts_with)
    assert prompt_contains in loaded.prompt_template


@pytest.mark.parametrize(
    "frame",
    [
        None,
        UserFlowStackFrame(flow_id="foo", step_id="first_step", frame_id="some-id"),
        ChitChatStackFrame(frame_id="foobar"),
    ],
)
def test_search_policy_does_not_support_other_frames(frame: DialogueStackFrame) -> None:
    assert not EnterpriseSearchPolicy.does_support_stack_frame(frame)


def test_search_policy_does_support_search_frame() -> None:
    frame = SearchStackFrame(
        frame_id="some-id",
    )
    assert EnterpriseSearchPolicy.does_support_stack_frame(frame)


@pytest.mark.parametrize(
    "dialogue_stack",
    [
        DialogueStack(frames=[]),
        DialogueStack(
            frames=[
                UserFlowStackFrame(
                    flow_id="foo", step_id="first_step", frame_id="some-id"
                )
            ]
        ),
        DialogueStack(
            frames=[
                UserFlowStackFrame(
                    flow_id="foo", step_id="first_step", frame_id="some-id"
                ),
                ChitChatStackFrame(frame_id="foobar"),
            ]
        ),
        DialogueStack(
            frames=[
                SearchStackFrame(frame_id="foobar"),
                UserFlowStackFrame(
                    flow_id="foo", step_id="first_step", frame_id="some-id"
                ),
            ]
        ),
    ],
)
def test_search_policy_abstains(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
    monkeypatch: MonkeyPatch,
    dialogue_stack: DialogueStack,
) -> None:
    """Test that the policy abstains with a stack that is not supported.

    Various dialogue stacks are tested to ensure that the policy does not predict
    anything when Search Stack frame is not at the top.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    domain = Domain.empty()

    # create a tracker with the stack set
    tracker = DialogueStateTracker.from_events(
        "test policy prediction",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen")],
    )
    tracker.update_stack(dialogue_stack)

    assert not default_enterprise_search_policy.supports_current_stack_frame(
        tracker=tracker
    )


def test_enterprise_search_policy_llm_config(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={
            "llm": {
                "model": "gpt-4",
                "request_timeout": 100,
                "max_tokens": 20,
            }
        },
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.config.get(LLM_CONFIG_KEY, {}).get("model") == "gpt-4"
    assert policy.config.get(LLM_CONFIG_KEY, {}).get("request_timeout") == 100
    assert policy.config.get(LLM_CONFIG_KEY, {}).get("max_tokens") == 20


def test_enterprise_search_policy_embeddings_config(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={
            "embeddings": {
                "type": "cohere",
                "model": "embed-english-v2.0",
            }
        },
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.config.get("embeddings", {}).get("type") == "cohere"
    assert policy.config.get("embeddings", {}).get("model") == "embed-english-v2.0"


def test_enterprise_search_policy_vector_store_config(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={
            "vector_store": {
                "type": "milvus",
            }
        },
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.vector_store_config.get("type") == "milvus"


def test_train_faiss_with_non_existing_documents_path(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    docs_dir = tmp_path / "non_existent_folder"
    assert not docs_dir.exists()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    with pytest.raises(EnterpriseSearchPolicyError) as exc_info:
        policy.train([], Domain.empty(), None, None, None)

    expected_error_code = (
        "core.policies.enterprise_search_policy.train.faiss.invalid_source_directory"
    )
    expected_error_msg_fragment = (
        "Document source directory does not exist or is not a directory"
    )

    err = exc_info.value
    assert err.code == expected_error_code
    assert expected_error_msg_fragment in str(err)


def test_train_faiss_with_invalid_documents_path(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    docs_dir = tmp_path / "existing_file.txt"
    docs_dir.touch()
    assert docs_dir.exists() and not docs_dir.is_dir()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    expected_error_code = (
        "core.policies.enterprise_search_policy.train" ".faiss.invalid_source_directory"
    )
    expected_msg_substring = (
        "Document source directory does not exist " "or is not a directory"
    )

    with pytest.raises(EnterpriseSearchPolicyError) as exc_info:
        policy.train([], Domain.empty(), None, None, None)

    err = exc_info.value
    assert err.code == expected_error_code
    assert expected_msg_substring in str(err)


def test_train_faiss_with_empty_documents_path(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    assert docs_dir.exists()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    expected_error_code = (
        "core.policies.enterprise_search_policy.train" ".faiss.source_directory_empty"
    )
    expected_msg_substring = "Document source directory is empty"

    with pytest.raises(EnterpriseSearchPolicyError) as exc_info:
        policy.train([], Domain.empty(), None, None, None)

    err = exc_info.value
    assert err.code == expected_error_code
    assert expected_msg_substring in str(err)


def test_train_faiss_with_valid_documents_path(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    docs_dir = tmp_path / "test_train_faiss_with_valid_documents_path"
    docs_dir.mkdir()
    assert docs_dir.exists() and docs_dir.is_dir()
    example_doc = docs_dir / "example.txt"
    example_doc.write_text("This is an example document.")
    assert example_doc.exists() and example_doc.is_file()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    try:
        policy._validate_documents_folder(docs_dir)
    except SystemExit:
        pytest.fail("SystemExit was raised unexpectedly")


def test_train_faiss_with_valid_documents_path_recursive_structure(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    docs_dir = tmp_path / "test_train_faiss_with_valid_documents_path"
    docs_dir.mkdir()

    nested_dir = docs_dir / "nested_dir"
    nested_dir.mkdir()

    example_doc = nested_dir / "example.txt"
    example_doc.write_text("This is an example document.")
    assert example_doc.exists() and example_doc.is_file()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    try:
        policy._validate_documents_folder(docs_dir)
    except SystemExit:
        pytest.fail("SystemExit was raised unexpectedly")


def test_enterprise_search_policy_fingerprint_addon_not_faiss_vector_store(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    config = {
        "vector_store": {
            "type": "milvus",
        }
    }
    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy._get_local_knowledge_data(store_type="milvus", source=None) is None


def test_enterprise_search_policy_fingerprint_addon_no_source_given(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    # Missing source property
    assert policy._get_local_knowledge_data(store_type="faiss") is None


def test_enterprise_search_policy_fingerprint_addon_faiss_no_file(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy._get_local_knowledge_data(config) is None


def test_enterprise_search_policy_fingerprint_addon_faiss_non_existing_source(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    assert not docs_dir.exists()

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy._get_local_knowledge_data(config) is None


def test_enterprise_search_policy_fingerprint_addon_faiss_with_files(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file = docs_dir / "doc1.txt"
    file.write_text("This is a test document.")

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.fingerprint_addon(config) is not None
    assert policy.fingerprint_addon(config) == policy.fingerprint_addon(config)


def test_enterprise_search_policy_fingerprint_addon_faiss_different_fingerprints(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file = docs_dir / "doc1.txt"
    file.write_text("This is a test document.")

    config = {
        "vector_store": {
            "type": "faiss",
            "source": str(docs_dir),
        }
    }

    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    fingerprint_1 = policy.fingerprint_addon(config)

    file.write_text("This is a test document. It has been changed.")

    fingerprint_2 = policy.fingerprint_addon(config)
    assert fingerprint_1 != fingerprint_2


def test_enterprise_search_policy_fingerprint_addon_diff_in_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    prompt_dir = Path(tmp_path) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "enterprise_search_policy_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {"prompt": str(prompt_file), "vector_store": {"type": "dummy"}}

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    fingerprint_1 = policy.fingerprint_addon(config)

    prompt_file.write_text("This is a test prompt. It has been changed.")

    fingerprint_2 = policy.fingerprint_addon(config)
    assert fingerprint_1 != fingerprint_2


def test_enterprise_search_policy_fingerprint_addon_no_diff_in_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    tmp_path: Path,
) -> None:
    prompt_dir = Path(tmp_path) / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompt_dir / "enterprise_search_policy_prompt.jinja2"
    prompt_file.write_text("This is a test prompt")

    config = {"prompt": str(prompt_file), "vector_store": {"type": "dummy"}}

    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    fingerprint_1 = policy.fingerprint_addon(config)
    fingerprint_2 = policy.fingerprint_addon(config)
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2


def test_enterprise_search_policy_fingerprint_addon_default_prompt_template(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    config = {"vector_store": {"type": "dummy"}}
    policy = EnterpriseSearchPolicy(
        {},
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    fingerprint_1 = policy.fingerprint_addon(config)
    fingerprint_2 = policy.fingerprint_addon(config)
    assert fingerprint_1 is not None
    assert fingerprint_1 == fingerprint_2


async def test_enterprise_search_policy_vector_store_config_error(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_internal_error: MagicMock,
) -> None:
    tracker = enterprise_search_tracker

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value
        # Mock _connect_vector_store_or_raise
        # to raise a VectorStoreConfigurationError
        with patch.object(
            mocked_enterprise_search_policy,
            "_connect_vector_store_or_raise",
            side_effect=VectorStoreConfigurationError("Mocked error"),
        ):
            await mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


async def test_enterprise_search_policy_vector_store_search_error(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_internal_error: MagicMock,
) -> None:
    tracker = enterprise_search_tracker

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value
        # Mock `self.vector_store.search(search_query)`
        # to raise Exception
        with patch.object(
            mocked_enterprise_search_policy.vector_store,
            "search",
            side_effect=InformationRetrievalException,
        ):
            await mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


async def test_enterprise_search_policy_none_llm_answer(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_internal_error: MagicMock,
) -> None:
    tracker = enterprise_search_tracker

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # mock self._invoke_llm(llm, prompt) to return None
        with patch.object(
            mocked_enterprise_search_policy,
            "_invoke_llm",
            return_value=None,
        ):
            await mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


async def test_enterprise_search_policy_no_retrieval(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_cannot_handle: MagicMock,
) -> None:
    tracker = enterprise_search_tracker
    search_results = SearchResultList(results=[], metadata={})

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # mock self.vector_store.search() to return empty results
        with patch.object(
            mocked_enterprise_search_policy.vector_store,
            "search",
            return_value=search_results,
        ):
            await mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            mock_create_prediction_cannot_handle.assert_called_once()


@pytest.mark.parametrize(
    "events,search_query",
    [
        ([UserUttered("search")], "search"),
        ([BotUttered("Hi, I am a bot")], "Hi, I am a bot"),
        ([UserUttered("\nsearch\n\nthis query")], " search  this query"),
        (
            [
                UserUttered("why is the sky blue?"),
                BotUttered("let me find out the answer for you..."),
            ],
            "let me find out the answer for you... why is the sky blue?",
        ),
        (
            [
                UserUttered("search"),
                BotUttered("first message after query..."),
                BotUttered("second message after query..."),
            ],
            "second message after query... first message after query...",
        ),
        (
            [
                BotUttered("Hi, I'm a bot."),
                BotUttered("Can I help you with something?"),
                UserUttered("why is the sky blue?"),
            ],
            "why is the sky blue? Can I help you with something?",
        ),
    ],
)
def test_prepare_search_query(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
    events: List,
    search_query: str,
) -> None:
    tracker = DialogueStateTracker.from_events(
        sender_id="test_policy_prediction",
        slots=[],
        evts=events,
    )

    assert (
        default_enterprise_search_policy._prepare_search_query(tracker, 2)
        == search_query
    )


def test_enterprise_search_policy_citation_enabled(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    expected_template = get_prompt_template(
        None, DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE
    )
    policy = EnterpriseSearchPolicy(
        config={**{"vector_store": {"type": "milvus"}, "citation_enabled": True}},
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    assert policy.citation_enabled is True
    assert policy.prompt_template == expected_template


def test_enterprise_search_policy_citation_disabled(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
) -> None:
    citation_prompt_template = get_prompt_template(
        None, DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE
    )
    assert default_enterprise_search_policy.citation_enabled is False
    assert default_enterprise_search_policy.prompt_template != citation_prompt_template


def test_enterprise_search_policy_post_process_citations_same_order(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that the citations are correctly re-ordered.

    The original LLM answer contains the citations in the sources list
    in the correct order with incorrect bracketed numbers.
    """
    first_source_citation = "3"
    second_source_citation = "1"
    third_source_citation = "2"

    llm_answer = f"""
This is a test answer with a citation [{first_source_citation}]. This is another test answer with a citation [{second_source_citation}]. This is a third test answer with a citation [{third_source_citation}].

Sources:

[{first_source_citation}] https://www.example.com/{first_source_citation}
[{second_source_citation}] https://www.example.com/{second_source_citation}
[{third_source_citation}] https://www.example.com/{third_source_citation}""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            f"""This is a test answer with a citation [1]. This is another test answer with a citation [2]. This is a third test answer with a citation [3].
Sources:
[1] https://www.example.com/{first_source_citation}
[2] https://www.example.com/{second_source_citation}
[3] https://www.example.com/{third_source_citation}"""  # noqa: E501
        ).strip()
    )


def test_enterprise_search_policy_post_process_citations_diff_order(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that the citations are correctly ordered.

    The original LLM answer contains the citations in the sources list
    in the incorrect order with incorrect bracketed numbers.
    """
    first_source_citation = "3"
    second_source_citation = "2"
    third_source_citation = "1"
    llm_answer = f"""
This is a test answer with a citation [{first_source_citation}]. This is another test answer with a citation [{second_source_citation}]. This is a third test answer with a citation [{third_source_citation}].

Sources:
[{first_source_citation}] https://www.example.com/{first_source_citation}
[{third_source_citation}] https://www.example.com/{third_source_citation}
[{second_source_citation}] https://www.example.com/{second_source_citation}""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            f"""This is a test answer with a citation [1]. This is another test answer with a citation [2]. This is a third test answer with a citation [3].
Sources:
[1] https://www.example.com/{first_source_citation}
[2] https://www.example.com/{second_source_citation}
[3] https://www.example.com/{third_source_citation}"""  # noqa: E501
        ).strip()
    )


@pytest.mark.parametrize(
    "llm_answer, expected_answer",
    [
        # Test that sources are returned as is when there are no relevant sources
        (
            """This is a test answer without relevant sources.

Sources:

No relevant sources.""",
            """This is a test answer without relevant sources.
Sources:
No relevant sources.""",
        ),
        # Test that sources are returned as is when there are no sources
        (
            "This is a test answer without sources.",
            "This is a test answer without sources.",
        ),
        # LLM answer contain multiple sources, but there is no citation.
        (
            """This is a test answer without a proper citation.

Sources:
[1] https://www.example.com/1
[2] https://www.example.com/2
[3] https://www.example.com/3""",
            """This is a test answer without a proper citation.
Sources:
[1] https://www.example.com/1
[2] https://www.example.com/2
[3] https://www.example.com/3""",
        ),
        # LLM answer contain multiple sources, but there is only one citation.
        (
            """This is a text with some citations [1].

Sources:
[1] example.org/abc
[2] example.org/def""",
            """This is a text with some citations [1].
Sources:
[1] example.org/abc
[2] example.org/def""",
        ),
        # LLM answer contain citations, but no sources
        (
            """Test answer with citations but without sources.[1][2]

Sources:
""",
            """Test answer with citations but without sources.""",
        ),
        # LLM answer contain multiple citations, but some of the sources are not present
        (
            """Test answer with a citations, but some sources are missing. [1][2][3]

Sources:
[1] https://www.example.com/abc
[3] https://www.example.com/ghi""",
            """Test answer with a citations, but some sources are missing. [1][2]
Sources:
[1] https://www.example.com/abc
[2] https://www.example.com/ghi""",
        ),
        # LLM answer contain multiple citations, but some of the sources are not present
        # and the order is wrong
        (
            """Test answer with a citations[3], but some sources are missing.[1][2]

Sources:
[1] https://www.example.com/abc
[3] https://www.example.com/ghi""",
            """Test answer with a citations[1], but some sources are missing.[2]
Sources:
[1] https://www.example.com/ghi
[2] https://www.example.com/abc""",
        ),
        # LLM answer contain multiple citations, but some of the sources are not present
        # and the order is wrong. Sources contain colons.
        (
            """Test answer with a citations[3], but some sources are missing.[1][2]

Sources:
[1]: https://www.example.com/abc
[3]: https://www.example.com/ghi""",
            """Test answer with a citations[1], but some sources are missing.[2]
Sources:
[1]: https://www.example.com/ghi
[2]: https://www.example.com/abc""",
        ),
        # Test that a citation group with duplicate numbers is correctly deduplicated
        # and sorted
        (
            """This is a test with a group citation [2, 1, 2].

Sources:
[1] https://www.example.com/abc
[2] https://www.example.com/def""",
            """This is a test with a group citation [1, 2].
Sources:
[1] https://www.example.com/def
[2] https://www.example.com/abc""",
        ),
        # Test that all citations are removed if none are valid, and sources are
        # preserved.
        (
            """Here is a citation[5] and another one [9]. Neither is valid.

Sources:
[1] Source 1
[2] Source 2""",
            """Here is a citation and another one. Neither is valid.
Sources:
[1] Source 1
[2] Source 2""",
        ),
    ],
)
def test_enterprise_search_policy_post_process_citations_parametrized(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    llm_answer: str,
    expected_answer: str,
) -> None:
    llm_answer = textwrap.dedent(llm_answer).strip()
    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)
    expected_answer = textwrap.dedent(expected_answer).strip()
    assert processed_answer == expected_answer


def test_enterprise_search_policy_post_process_citations_consecutive_citations(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that consecutive citations are correctly ordered."""
    first_source_citation = "3"
    second_source_citation = "2"
    third_source_citation = "1"
    llm_answer = f"""
This is a test answer with a citation [{first_source_citation}][{second_source_citation}]. This is another test answer with a citation [{third_source_citation}].

Sources:
[{first_source_citation}] https://www.example.com/{first_source_citation}
[{third_source_citation}] https://www.example.com/{third_source_citation}
[{second_source_citation}] https://www.example.com/{second_source_citation}""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            f"""This is a test answer with a citation [1][2]. This is another test answer with a citation [3].
Sources:
[1] https://www.example.com/{first_source_citation}
[2] https://www.example.com/{second_source_citation}
[3] https://www.example.com/{third_source_citation}"""  # noqa: E501
        ).strip()
    )


@pytest.mark.parametrize("separator", [", ", ","])
def test_enterprise_search_policy_post_process_citations_nested_citations(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    separator: str,
) -> None:
    """Test that nested citations are correctly ordered."""
    first_source_citation = "3"
    second_source_citation = "1"
    third_source_citation = "2"
    llm_answer = f"""
This is a test answer with a citation [{first_source_citation}{separator}{second_source_citation}]. This is another test answer with a citation [{third_source_citation}].

Sources:
[{first_source_citation}] https://www.example.com/{first_source_citation}
[{second_source_citation}] https://www.example.com/{second_source_citation}
[{third_source_citation}] https://www.example.com/{third_source_citation}""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            f"""This is a test answer with a citation [1, 2]. This is another test answer with a citation [3].
Sources:
[1] https://www.example.com/{first_source_citation}
[2] https://www.example.com/{second_source_citation}
[3] https://www.example.com/{third_source_citation}"""  # noqa: E501
        ).strip()
    )


@pytest.mark.parametrize("separator", [", ", ","])
def test_enterprise_search_policy_post_process_citations_multiple_nested_citations(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    separator: str,
) -> None:
    """Test that nested citations are correctly ordered."""
    first_citation = "3"
    second_citation = "1"
    third_citation = "4"
    fourth_citation = "2"
    llm_answer = f"""
This is a test answer with a citation [{first_citation}{separator}{second_citation}{separator}{third_citation}]. This is another test answer with a citation [{fourth_citation}].

Sources:
[{first_citation}] https://www.example.com/{first_citation}
[{second_citation}] https://www.example.com/{second_citation}
[{third_citation}] https://www.example.com/{third_citation}
[{fourth_citation}] https://www.example.com/{fourth_citation}""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            f"""This is a test answer with a citation [1, 2, 3]. This is another test answer with a citation [4].
Sources:
[1] https://www.example.com/{first_citation}
[2] https://www.example.com/{second_citation}
[3] https://www.example.com/{third_citation}
[4] https://www.example.com/{fourth_citation}"""  # noqa: E501
        ).strip()
    )


def test_enterprise_search_policy_post_process_citations_with_numbers_in_llm_answer(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that numbers in the llm answer are not matched as citation indices."""
    number = "136"
    llm_answer = f"""
You can find directions to campus by following PA Route {number} West, turning left onto College Street, then left at the next stoplight onto Wheeling Street. Continue straight down the hill to the Burnett Center on your right, then turn right onto Grant Street. The Taylor lot will be on your left [1].
Sources:
[1] docs/txt/52a4386a.txt""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert processed_answer.strip() == llm_answer


def test_enterprise_search_policy_post_process_citations_numbers_identical_to_source_indices(  # noqa: E501
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that numbers in the llm answer are not matched as citation indices.

    The number in the llm answer is identical to the source index.
    """
    number = "2"
    llm_answer = f"""
You can find directions to campus by following PA Route {number} West, turning left onto College Street, then left at the next stoplight onto Wheeling Street. Continue straight down the hill to the Burnett Center on your right, then turn right onto Grant Street. The Taylor lot will be on your left [2].

Sources:

[2] docs/txt/52a4386a.txt""".strip()  # noqa: E501

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == f"""
You can find directions to campus by following PA Route {number} West, turning left onto College Street, then left at the next stoplight onto Wheeling Street. Continue straight down the hill to the Burnett Center on your right, then turn right onto Grant Street. The Taylor lot will be on your left [1].
Sources:
[1] docs/txt/52a4386a.txt""".strip()  # noqa: E501
    )


def test_enterprise_search_policy_check_relevancy_enabled_but_generative_search_is_disabled(  # noqa: E501
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    # Given
    expected_event = (
        "enterprise_search_policy"
        ".relevancy_check_enabled_with_disabled_generative_search"
    )
    expected_log_level = "warning"
    expected_log_message_parts = [
        f"The config parameter '{CHECK_RELEVANCY_PROPERTY}' is set to"
        "'True', but the generative search is disabled"
    ]
    expected_prompt_template = get_prompt_template(
        None,
        DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_RELEVANCY_CHECK_AND_CITATION_TEMPLATE,
    )

    # When
    with structlog.testing.capture_logs() as caplog:
        policy = EnterpriseSearchPolicy(
            config={
                **{
                    "vector_store": {"type": "milvus"},
                    "check_relevancy": True,
                    "use_generative_llm": False,
                }
            },
            model_storage=default_model_storage,
            resource=Resource("enterprise_search_policy"),
            execution_context=default_execution_context,
            vector_store=vector_store,
        )
        logs = filter_logs(
            caplog, expected_event, expected_log_level, expected_log_message_parts
        )

    # Then
    assert policy.relevancy_check_enabled is True
    assert policy.use_llm is False
    assert policy.prompt_template == expected_prompt_template
    assert len(logs) == 1


def test_enterprise_search_policy_check_relevancy_enabled(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    expected_prompt_template = get_prompt_template(
        None,
        DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_RELEVANCY_CHECK_AND_CITATION_TEMPLATE,
    )
    policy = EnterpriseSearchPolicy(
        config={**{"vector_store": {"type": "milvus"}, "check_relevancy": True}},
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    assert policy.relevancy_check_enabled is True
    assert policy.prompt_template == expected_prompt_template


def test_enterprise_search_policy_check_relevancy_disabled(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
) -> None:
    expected_prompt_template = get_prompt_template(
        None, DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE
    )
    assert default_enterprise_search_policy.relevancy_check_enabled is False
    assert default_enterprise_search_policy.prompt_template == expected_prompt_template


async def test_enterprise_search_policy_tracker_state_is_passed(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
) -> None:
    tracker = enterprise_search_tracker
    search_results = SearchResultList(results=[], metadata={})

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # assert self.vector_store.search was called with tracker_state
        with patch.object(
            mocked_enterprise_search_policy.vector_store,
            "search",
            return_value=search_results,
        ) as mock_search:
            await mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            mock_search.assert_called_once_with(
                query="what is the meaning of life?",
                tracker_state=tracker.current_state(EventVerbosity.AFTER_RESTART),
                threshold=0.0,
            )


def test_enterprise_search_policy_use_llm_config(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={
            USE_LLM_PROPERTY: False,
        },
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy.config.get(USE_LLM_PROPERTY) is False


async def test_enterprise_search_policy_response_with_use_llm_false(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    enterprise_search_tracker: DialogueStateTracker,
    search_results: SearchResultList,
    monkeypatch: MonkeyPatch,
) -> None:
    """Given the `USE_LLM_PROPERTY` is set to False, the policy should return
    a response without using the LLM. Response text should be from the first
    search result.
    """
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    policy = EnterpriseSearchPolicy(
        config={USE_LLM_PROPERTY: False},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # mock self.vector_store.search() to return search results
        with patch.object(
            policy.vector_store,
            "search",
            return_value=search_results,
        ):
            prediction = await policy.predict_action_probabilities(
                tracker=enterprise_search_tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            assert isinstance(prediction, PolicyPrediction)
            assert (
                prediction.action_metadata.get("message").get("text") == "test response"
            )


async def test_enterprise_search_policy_response_with_use_llm_true(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    enterprise_search_tracker: DialogueStateTracker,
    search_results: SearchResultList,
    monkeypatch: MonkeyPatch,
    llm_response_object: LLMResponse,
) -> None:
    """Given the `USE_LLM_PROPERTY` is set to True, the policy should return
    a response using the LLM. Response text should be from the LLM.
    """
    monkeypatch.setenv(OPENAI_API_KEY_ENV_VAR, "my key")
    policy = EnterpriseSearchPolicy(
        config={USE_LLM_PROPERTY: True},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # mock self.vector_store.search() to return search results
        with patch.object(
            policy.vector_store,
            "search",
            return_value=search_results,
        ):
            # mock self._invoke_llm(prompt) to return LLM generated response
            llm_response_object.choices = ["LLM generated response"]
            with patch.object(
                policy,
                "_invoke_llm",
                return_value=llm_response_object,
            ):
                prediction = await policy.predict_action_probabilities(
                    tracker=enterprise_search_tracker,
                    domain=Domain.empty(),
                    endpoints=None,
                )

                assert isinstance(prediction, PolicyPrediction)

                message_metadata = prediction.action_metadata.get("message")
                assert message_metadata.get("text") == "LLM generated response"
                assert (
                    message_metadata.get(UTTER_SOURCE_METADATA_KEY)
                    == "EnterpriseSearchPolicy"
                )
                assert SEARCH_QUERY_METADATA_KEY in message_metadata
                assert message_metadata.get(SEARCH_RESULTS_METADATA_KEY) == [
                    result.text for result in search_results.results
                ]


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
    default_enterprise_search_policy: EnterpriseSearchPolicy,
):
    tracker = DialogueStateTracker(
        "id1",
        slots=[BooleanSlot(ROUTE_TO_CALM_SLOT, [], initial_value=routing_slot_value)],
    )

    assert result == default_enterprise_search_policy.should_abstain_in_coexistence(
        tracker, True
    )


@pytest.mark.parametrize(
    "config, expected_llm_config, expected_embedding_config",
    [
        (
            {
                LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
                EMBEDDINGS_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
            },
            {"provider": "openai", "model": "gpt-4"},
            {"provider": "openai", "model": "gpt-4"},
        ),
        (
            {
                "user_input": {"max_characters": -1},
            },
            DEFAULT_LLM_CONFIG,
            DEFAULT_EMBEDDINGS_CONFIG,
        ),
        (
            {
                LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
            },
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
        ),
        (
            {
                LLM_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
            },
            {"provider": "openai", "model": "gpt-4"},
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
        ),
        (
            {
                LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-4"},
                EMBEDDINGS_CONFIG_KEY: {"provider": "openai", "model": "gpt-4"},
            },
            {
                "id": "openai_gpt-4",
                "models": [{"provider": "openai", "model": "gpt-4"}],
            },
            {"provider": "openai", "model": "gpt-4"},
        ),
    ],
)
def test_enterprise_search_policy_init_with_different_llm_configs(
    config: Optional[Dict[str, Any]],
    expected_llm_config: Optional[Dict[str, Any]],
    expected_embedding_config: Optional[Dict[str, Any]],
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource: Resource,
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch,
) -> None:
    mock_available_endpoints.model_groups = [
        {
            "id": "openai_gpt-4",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        },
        {
            "id": "openai_embedding",
            "models": [{"provider": "openai", "model": "text-embedding-3-large"}],
        },
    ]

    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    policy = EnterpriseSearchPolicy(
        config, default_model_storage, resource, default_execution_context
    )
    assert policy.llm_config == expected_llm_config
    assert policy.embeddings_config == expected_embedding_config


def test_enterprise_search_policy_persist_config(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    resource: Resource,
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_available_endpoints.model_groups = [
        {
            "id": "model_group_id",
            "models": [{"provider": "openai", "model": "gpt-4"}],
        }
    ]

    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    config = {
        LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "model_group_id"},
        EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "model_group_id"},
    }
    component = EnterpriseSearchPolicy(
        config, default_model_storage, resource, default_execution_context
    )

    # Ensure the config is resolved
    assert component.llm_config == {
        "id": "model_group_id",
        "models": [{"provider": "openai", "model": "gpt-4"}],
    }
    assert component.embeddings_config == {
        "id": "model_group_id",
        "models": [{"provider": "openai", "model": "gpt-4"}],
    }

    # Persist the generator
    component.persist()

    # Check that the persisted config is equal to our config
    with default_model_storage.read_from(resource) as path:
        persisted_config = rasa.shared.utils.io.read_json_file(
            path / ENTERPRISE_SEARCH_CONFIG_FILE_NAME
        )

    assert persisted_config[LLM_CONFIG_KEY] == {
        "id": "model_group_id",
        "models": [{"provider": "openai", "model": "gpt-4"}],
    }
    assert persisted_config[EMBEDDINGS_CONFIG_KEY] == {
        "id": "model_group_id",
        "models": [{"provider": "openai", "model": "gpt-4"}],
    }


@pytest.mark.parametrize(
    "config_1, model_groups_1, config_2, model_groups_2, fingerprint_differs",
    [
        (
            {"vector_store": {"type": "custom_vector_store"}},
            [],
            {"vector_store": {"type": "custom_vector_store"}},
            [],
            False,
        ),
        (
            {
                LLM_CONFIG_KEY: {
                    MODEL_GROUP_CONFIG_KEY: "openai_gpt",
                },
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_gpt",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
            ],
            {
                LLM_CONFIG_KEY: {
                    MODEL_GROUP_CONFIG_KEY: "openai_gpt",
                },
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_gpt",
                    "models": [
                        {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"}
                    ],
                },
            ],
            True,
        ),
        (
            {
                LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-1"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_gpt-1",
                    "models": [{"provider": "openai", "model": "gpt-4"}],
                },
            ],
            {
                LLM_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_gpt-2"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_gpt-2",
                    "models": [
                        {"provider": "openai", "model": "gpt-4.1-mini-2025-04-14"}
                    ],
                },
            ],
            True,
        ),
        (
            {
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_embeddings"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_embeddings",
                    "models": [{"provider": "openai", "model": "embedding-model-1"}],
                },
            ],
            {
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_embeddings"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_embeddings",
                    "models": [{"provider": "openai", "model": "embedding-model-2"}],
                },
            ],
            True,
        ),
        (
            {
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_embeddings-1"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_embeddings-1",
                    "models": [{"provider": "openai", "model": "embedding-model"}],
                },
            ],
            {
                EMBEDDINGS_CONFIG_KEY: {MODEL_GROUP_CONFIG_KEY: "openai_embeddings-2"},
                "vector_store": {"type": "custom_vector_store"},
            },
            [
                {
                    "id": "openai_embeddings-2",
                    "models": [{"provider": "openai", "model": "embedding-model"}],
                },
            ],
            True,
        ),
    ],
)
async def test_enterprise_search_policy_fingerprint_addon_with_different_model_configs(
    config_1: Dict[str, Any],
    model_groups_1: List[Dict[str, Any]],
    config_2: Dict[str, Any],
    model_groups_2: List[Dict[str, Any]],
    fingerprint_differs: bool,
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    mock_available_endpoints: MagicMock,
    mock_configuration: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = EnterpriseSearchPolicy(
        config={"vector_store": {"type": "custom_vector_store"}},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    monkeypatch.setattr("rasa.shared.utils.llm.Configuration", mock_configuration)

    mock_available_endpoints.model_groups = model_groups_1
    fingerprint_1 = policy.fingerprint_addon(config_1)

    mock_available_endpoints.model_groups = model_groups_2
    fingerprint_2 = policy.fingerprint_addon(config_2)

    assert fingerprint_1 is not None
    assert fingerprint_2 is not None
    if fingerprint_differs:
        assert fingerprint_1 != fingerprint_2
    else:
        assert fingerprint_1 == fingerprint_2


def test_add_prompt_and_llm_response_to_latest_message_with_llm_response(
    llm_response_object: LLMResponse,
):
    tracker = DialogueStateTracker("default", slots={})
    tracker.update(UserUttered("Hello"))
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"

    with set_record_commands_and_prompts():
        EnterpriseSearchPolicy._add_prompt_and_llm_response_to_latest_message(
            tracker, prompt_name, user_prompt, llm_response_object
        )

    parse_data = tracker.latest_message.parse_data
    assert parse_data[PROMPTS] == [
        {
            KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: llm_response_object.to_dict(),
        },
    ]


def test_add_prompt_and_llm_response_to_latest_message_without_llm_response():
    tracker = DialogueStateTracker("default", slots={})
    tracker.update(UserUttered("Hello"))
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"

    with set_record_commands_and_prompts():
        EnterpriseSearchPolicy._add_prompt_and_llm_response_to_latest_message(
            tracker, prompt_name, user_prompt
        )

    parse_data = tracker.latest_message.parse_data
    assert parse_data[PROMPTS] == [
        {
            KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: None,
        }
    ]


def test_add_prompt_and_llm_response_to_latest_message_existing_prompts():
    tracker = DialogueStateTracker("default", slots={})
    tracker.update(UserUttered("Hello"))
    tracker.latest_message.parse_data = {
        PROMPTS: [
            {
                KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
                KEY_PROMPT_NAME: "existing_prompt",
                KEY_USER_PROMPT: "Existing prompt",
            }
        ]
    }
    prompt_name = "test_prompt"
    user_prompt = "What is the weather like?"

    with set_record_commands_and_prompts():
        EnterpriseSearchPolicy._add_prompt_and_llm_response_to_latest_message(
            tracker, prompt_name, user_prompt
        )

    parse_data = tracker.latest_message.parse_data
    assert parse_data[PROMPTS] == [
        {
            KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
            KEY_PROMPT_NAME: "existing_prompt",
            KEY_USER_PROMPT: "Existing prompt",
        },
        {
            KEY_COMPONENT_NAME: EnterpriseSearchPolicy.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: None,
        },
    ]


@pytest.mark.parametrize(
    "documents",
    [
        [
            SearchResult(metadata="Document 1", text="This is the first document."),
            SearchResult(metadata="Document 2", text="This is the second document."),
        ],
        [
            SearchResult(metadata="Doc A", text="Content of A."),
            SearchResult(metadata="Doc B", text="Content of B."),
        ],
    ],
)
def test_render_prompt_includes_doc_text(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
    documents: List[SearchResult],
    monkeypatch: MonkeyPatch,
):
    """Test that _render_prompt correctly includes doc.text in the rendered output."""
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    domain = Domain.empty()

    tracker = DialogueStateTracker.from_events(
        "test render prompt",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen")],
    )

    rendered_prompt = default_enterprise_search_policy._render_prompt(
        tracker, documents
    )

    for doc in documents:
        assert doc.text in rendered_prompt


@pytest.mark.parametrize(
    "relevancy_check_enabled, citation_enabled, llm_answer, expected_text, expect_cannot_handle",  # noqa: E501
    [
        # Relevancy check enabled, generated answer
        (
            True,
            True,
            "Generated answer",
            "Generated answer - Relevancy - Citations",
            False,
        ),
        (True, False, "Generated answer", "Generated answer - Relevancy", False),
        # Relevancy check enabled but answer not relevant
        (True, True, "[NO_RAG_ANSWER]", None, True),
        (True, False, "[NO_RAG_ANSWER]", None, True),
        # Relevancy check disabled, generated answer
        (False, True, "Generated answer", "Generated answer - Citations", False),
        (False, False, "Generated answer", "Generated answer", False),
    ],
)
@patch("rasa.shared.utils.llm.llm_factory")
@patch.object(
    EnterpriseSearchPolicy,
    "_invoke_llm",
)
@patch.object(
    EnterpriseSearchPolicy,
    "post_process_citations",
)
async def test_enterprise_search_policy_prediction_varied_configs(
    mock_post_process_citations: MagicMock,
    mock_invoke_llm: MagicMock,
    mock_llm_factory: MagicMock,
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_cannot_handle: MagicMock,
    relevancy_check_enabled: bool,
    citation_enabled: bool,
    llm_answer: str,
    expected_text: Optional[str],
    expect_cannot_handle: bool,
) -> None:
    def simulate_citation_output() -> Optional[str]:
        """Simulate the output of the citation step depending on:
        - whether the citation step is enabled or not,
        - whether the relevancy check is enabled or not,
        """
        if llm_answer == "[NO_RELEVANT_ANSWER_FOUND]" or not citation_enabled:
            return None
        return mock_invoke_llm.return_value.choices[0] + " - Citations"

    def llm_answer_generation_output() -> Optional[str]:
        """Simulate the output of the LLM answer generation step depending on
        whether the relevancy check is enabled or not
        """
        if llm_answer == "[NO_RELEVANT_ANSWER_FOUND]" or not relevancy_check_enabled:
            return llm_answer
        return f"{llm_answer} - Relevancy"

    # Given
    mock_llm_factory.return_value = MagicMock()

    mocked_enterprise_search_policy.relevancy_check_enabled = relevancy_check_enabled
    mocked_enterprise_search_policy.citation_enabled = citation_enabled

    mock_invoke_llm.return_value = LLMResponse(
        id="test_response",
        choices=[llm_answer_generation_output()],
        created=123,
    )
    mock_post_process_citations.return_value = simulate_citation_output()

    domain = Domain.empty()
    tracker = enterprise_search_tracker

    # When
    prediction = await mocked_enterprise_search_policy.predict_action_probabilities(
        tracker=tracker,
        domain=domain,
        endpoints=None,
    )

    # Then
    mock_invoke_llm.assert_called_once()

    if expect_cannot_handle:
        mock_create_prediction_cannot_handle.assert_called_once_with(
            domain,
            tracker,
            RASA_PATTERN_CANNOT_HANDLE_NO_RELEVANT_ANSWER,
        )
        mock_post_process_citations.assert_not_called()
    else:
        mock_create_prediction_cannot_handle.assert_not_called()
        if citation_enabled:
            mock_post_process_citations.assert_called_once()
        else:
            mock_post_process_citations.assert_not_called()

        assert prediction.action_metadata["message"]["text"] == expected_text


@pytest.mark.parametrize(
    "config, prompt_starts_with, prompt_contains",
    [
        (
            {},
            "Given the following information, please provide an answer based on"
            " the provided documents",
            [],
        ),
        (
            {"citation_enabled": True},
            "Given the following information, please provide an answer based on"
            " the provided documents",
            ["Citing Sources"],
        ),
        (
            {"check_relevancy": True},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            ["[NO_RAG_ANSWER]"],
        ),
        (
            {"check_relevancy": True, "citation_enabled": True},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            [
                "[NO_RAG_ANSWER]",
                "Citing Sources",
            ],
        ),
        (
            {"check_relevancy": True, "citation_enabled": False},
            "{% if check_relevancy %}Based on the provided documents and the recent "
            "conversation context, answer the following question.",
            ["[NO_RAG_ANSWER]"],
        ),
    ],
)
def test_get_system_default_prompt_based_on_config(
    config: Dict[str, Any],
    prompt_starts_with: str,
    prompt_contains: List[str],
):
    # When
    prompt = EnterpriseSearchPolicy.get_system_default_prompt_based_on_config(config)

    # Then
    assert prompt.startswith(prompt_starts_with)
    for prompt_contains_str in prompt_contains:
        assert prompt_contains_str in prompt


@pytest.mark.parametrize(
    "use_generative_llm_config, expected_parse_as_faq_pairs",
    [
        ({USE_LLM_PROPERTY: True}, False),
        ({USE_LLM_PROPERTY: False}, True),
        ({}, not DEFAULT_USE_LLM_PROPERTY),
    ],
)
@patch("rasa.core.policies.enterprise_search_policy" ".FAISS_Store")
@patch(
    "rasa.core.policies.enterprise_search_policy"
    ".EnterpriseSearchPolicy._perform_health_checks"
)
@patch(
    "rasa.core.policies.enterprise_search_policy"
    ".track_enterprise_search_policy_train_started"
)
@patch(
    "rasa.core.policies.enterprise_search_policy"
    ".track_enterprise_search_policy_train_completed"
)
@patch(
    "rasa.core.policies.enterprise_search_policy"
    ".EnterpriseSearchPolicy._create_plain_embedder"
)
def test_train_and_load_calls_faiss_store_with_parsed_faq_when_use_generative_llm_is_disabled(  # noqa: E501
    mock_create_plain_embedder: Mock,
    mock_track_enterprise_search_policy_train_completed: Mock,
    mock_track_enterprise_search_policy_train_started: Mock,
    mock_perform_llm_health_check: Mock,
    mock_faiss_store,
    default_model_storage: ModelStorage,
    resource: Resource,
    use_generative_llm_config: dict,
    expected_parse_as_faq_pairs: bool,
    tmp_path: Path,
):
    docs_dir = tmp_path / "test_train_faiss_with_valid_documents_path"
    docs_dir.mkdir()
    assert docs_dir.exists() and docs_dir.is_dir()
    example_doc = docs_dir / "example.txt"
    example_doc.write_text("This is an example document.")
    assert example_doc.exists() and example_doc.is_file()

    # Given
    mock_create_plain_embedder.return_value = Mock()
    config = {
        "vector_store": {"type": "faiss", "source": str(docs_dir)},
        **use_generative_llm_config,
    }
    policy = EnterpriseSearchPolicy(
        config=config,
        model_storage=default_model_storage,
        resource=resource,
        execution_context=Mock(),
    )

    # When trained + loaded
    resource = policy.train(Mock(), Mock(), Mock(), Mock(), Mock())
    EnterpriseSearchPolicy.load(config, default_model_storage, resource, Mock())

    # Then
    # Once during training and once during loading
    assert mock_faiss_store.call_count == 2

    for call_args in mock_faiss_store.call_args_list:
        kwargs = call_args.kwargs
        assert kwargs["parse_as_faq_pairs"] is expected_parse_as_faq_pairs
