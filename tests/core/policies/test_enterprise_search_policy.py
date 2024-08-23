import textwrap
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms.fake import FakeListLLM
from pytest import MonkeyPatch

from rasa.core.constants import UTTER_SOURCE_METADATA_KEY

from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    ChitChatStackFrame,
    DialogueStackFrame,
    SearchStackFrame,
    UserFlowStackFrame,
)
from rasa.core.policies.policy import PolicyPrediction
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import OPENAI_API_KEY_ENV_VAR, LLM_CONFIG_KEY
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted, UserUttered, BotUttered
from rasa.shared.core.slots import BooleanSlot
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.core.information_retrieval import (
    InformationRetrieval,
    SearchResultList,
    SearchResult,
    InformationRetrievalException,
)
from rasa.core.policies.enterprise_search_policy import (
    SEARCH_QUERY_METADATA_KEY,
    SEARCH_RESULTS_METADATA_KEY,
    USE_LLM_PROPERTY,
    EnterpriseSearchPolicy,
    VectorStoreConfigurationError,
)


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
    monkeypatch.setenv("OPENAI_API_KEY", "test")
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
        (
            {"prompt": "data/prompt_templates/test_prompt.jinja2"},
            "Identify the user's message intent",
            "",
        ),
        (
            {},
            "Given the following information, please provide an answer based on"
            " the provided documents",
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
            {"citation_enabled": True},
            "Given the following information, please provide an answer based on"
            " the provided documents",
            "Citing Sources",
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
) -> None:
    """Test that the prompt is set correctly based on the config."""
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
    assert policy._get_local_knowledge_data(config) is None


def test_enterprise_search_policy_fingerprint_addon_no_source_given(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    # missing source property
    config = {"vector_store": {"type": "faiss"}}

    policy = EnterpriseSearchPolicy(
        config={},
        model_storage=default_model_storage,
        resource=Resource("enterprisesearchpolicy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )
    assert policy._get_local_knowledge_data(config) is None


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

        # mock self._generate_llm_answer(llm, prompt) to return None
        with patch.object(
            mocked_enterprise_search_policy,
            "_generate_llm_answer",
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
    policy = EnterpriseSearchPolicy(
        config={**{"vector_store": {"type": "milvus"}, "citation_enabled": True}},
        model_storage=default_model_storage,
        resource=Resource("enterprise_search_policy"),
        execution_context=default_execution_context,
        vector_store=vector_store,
    )

    assert policy.citation_enabled is True
    assert policy.prompt_template == policy.citation_prompt_template


def test_enterprise_search_policy_citation_disabled(
    default_enterprise_search_policy: EnterpriseSearchPolicy,
) -> None:
    assert default_enterprise_search_policy.citation_enabled is False
    assert (
        default_enterprise_search_policy.prompt_template
        != default_enterprise_search_policy.citation_prompt_template
    )


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


def test_enterprise_search_policy_post_process_citations_diff_format(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that sources are returned as is when there are no relevant sources."""
    llm_answer = """This is a test answer without relevant sources.

Sources:

No relevant sources.""".strip()

    llm_answer = "\n".join([line.rstrip() for line in llm_answer.splitlines()])

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)

    assert (
        processed_answer.strip()
        == textwrap.dedent(
            """This is a test answer without relevant sources.
Sources:
No relevant sources."""
        ).strip()
    )


def test_enterprise_search_policy_post_process_citations_no_sources(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
) -> None:
    """Test that the llm answer is returned as is when no sources are present."""
    llm_answer = "This is a test answer without sources."

    processed_answer = EnterpriseSearchPolicy.post_process_citations(llm_answer)
    assert processed_answer == llm_answer


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
            # mock self._generate_llm_answer(llm, prompt) to
            # return LLM generated response
            with patch.object(
                policy,
                "_generate_llm_answer",
                return_value="LLM generated response",
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
