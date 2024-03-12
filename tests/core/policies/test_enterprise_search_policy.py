from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.embeddings import FakeEmbeddings
from langchain.llms.fake import FakeListLLM
from pytest import MonkeyPatch
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    ChitChatStackFrame,
    DialogueStackFrame,
    SearchStackFrame,
    UserFlowStackFrame,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActionExecuted
from rasa.shared.core.trackers import DialogueStateTracker

from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrieval,
    InformationRetrievalException,
)
from rasa.core.policies.enterprise_search_policy import (
    LLM_CONFIG_KEY,
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
        "test policy prediction",
        domain=domain,
        slots=domain.slots,
        evts=[ActionExecuted(action_name="action_listen")],
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
    )
    policy.vector_store = vector_store
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


@pytest.mark.parametrize(
    "config,prompt_starts_with",
    [
        (
            {"prompt": "data/prompt_templates/test_prompt.jinja2"},
            "Identify the user's message intent",
        ),
        (
            {},
            "Given the following information, please provide an answer based on"
            " the provided documents",
        ),
    ],
)
async def test_enterprise_search_policy_prompt(
    default_model_storage: ModelStorage,
    default_execution_context: ExecutionContext,
    vector_store: InformationRetrieval,
    config: dict,
    prompt_starts_with: str,
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


def test_enterprise_search_policy_vector_store_config_error(
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
            mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


def test_enterprise_search_policy_vector_store_search_error(
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
            mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


def test_enterprise_search_policy_none_llm_answer(
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
            mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            # assert _create_prediction_internal_error was called
            mock_create_prediction_internal_error.assert_called_once()


def test_enterprise_search_policy_no_retrieval(
    mocked_enterprise_search_policy: EnterpriseSearchPolicy,
    enterprise_search_tracker: DialogueStateTracker,
    mock_create_prediction_cannot_handle: MagicMock,
) -> None:
    tracker = enterprise_search_tracker

    with patch("rasa.shared.utils.llm.llm_factory") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm_factory.return_value = mock_llm.return_value

        # mock self.vector_store.search() to return []
        with patch.object(
            mocked_enterprise_search_policy.vector_store,
            "search",
            return_value=[],
        ):
            mocked_enterprise_search_policy.predict_action_probabilities(
                tracker=tracker,
                domain=Domain.empty(),
                endpoints=None,
            )

            mock_create_prediction_cannot_handle.assert_called_once()
