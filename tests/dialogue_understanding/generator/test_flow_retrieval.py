import uuid
from typing import List, Text, Any, Dict
from unittest.mock import Mock, patch

import pytest
from pytest import MonkeyPatch
from _pytest.tmpdir import TempPathFactory
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from rasa.dialogue_understanding.generator.flow_retrieval import (
    FlowRetrieval,
    SHOULD_EMBED_SLOTS_KEY,
    MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY,
    TURNS_TO_EMBED_KEY,
    DEFAULT_TURNS_TO_EMBED,
    DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH,
)
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import EMBEDDINGS_CONFIG_KEY
from rasa.shared.core.events import (
    UserUttered,
    BotUttered,
    FlowStarted,
    FlowCompleted,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.slots import TextSlot, BooleanSlot, CategoricalSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.embedding._langchain_embedding_client_adapter import (
    _LangchainEmbeddingClientAdapter,
)
from rasa.shared.providers.embedding.openai_embedding_client import (
    OpenAIEmbeddingClient,
)
from rasa.shared.utils.llm import USER, AI


class TestFlowRetrieval:
    @pytest.fixture(scope="session")
    def flows(self) -> FlowsList:
        return flows_from_str(
            """
            flows:
                test_always_included_flow:
                    always_include_in_prompt: true
                    name: always included flow
                    description: a flow that is always included in the prompt
                    steps:
                    - id: first_step_a
                      action: action_listen

                test_link_based_flow:
                    if: False
                    name: link based flow
                    description: a test flow that is never included in the prompt
                    steps:
                    - id: first_step_b
                      action: action_listen

                test_flow_with_collect_steps:
                    name: flow with collect steps
                    description: test flow with multiple collect steps
                    steps:
                        - id: first_step
                          collect: test_slot_1
                          description: "A description for the first test slot"
                        - id: second_step
                          collect: test_slot_2
                          description: "A description for the second test slot"
                        - id: third_step
                          collect: test_slot_3
                          description: "A description for the third test slot"

            """
        )

    @pytest.fixture(scope="session")
    def domain(self) -> Mock:
        domain = Mock()
        test_slot_1 = TextSlot(
            name="test_slot_1",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_2 = BooleanSlot(
            name="test_slot_2",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
        )
        test_slot_3 = CategoricalSlot(
            name="test_slot_3",
            mappings=[{}],
            initial_value=None,
            influence_conversation=False,
            values=["A", "B", "C"],
        )
        domain.slots = [test_slot_1, test_slot_2, test_slot_3]
        return domain

    @pytest.fixture(scope="session")
    def startable_flows_documents(self) -> List[Document]:
        """
        Documents based on the flows fixture defined above
        """
        docs = [
            Document(
                page_content=(
                    "always included flow: a flow that is always included in the prompt"
                ),
                metadata={"flow_id": "test_always_included_flow"},
            ),
            Document(
                page_content=(
                    "flow with collect steps: test flow with multiple collect steps\n"
                    "    test_slot_1: A description for the first test slot\n"
                    "    test_slot_2: A description for the second test slot, allowed "
                    "values: [True, False]\n"
                    "    test_slot_3: A description for the third test slot, "
                    "allowed values: ['A', 'B', 'C']"
                ),
                metadata={"flow_id": "test_flow_with_collect_steps"},
            ),
        ]
        return docs

    @pytest.fixture(scope="session")
    def model_storage(self, tmp_path_factory: TempPathFactory) -> ModelStorage:
        return LocalModelStorage(tmp_path_factory.mktemp(uuid.uuid4().hex))

    @pytest.fixture(scope="session")
    def resource(self) -> Resource:
        return Resource(uuid.uuid4().hex)

    @pytest.fixture(scope="function")
    def flow_search(
        self, model_storage: ModelStorage, resource: Resource
    ) -> FlowRetrieval:
        return FlowRetrieval({}, model_storage, resource)

    def test_init(self, model_storage: ModelStorage, resource: Resource):
        # Given
        config = {}
        # When
        flow_search = FlowRetrieval(config, model_storage, resource)
        # Then
        assert flow_search.config == flow_search.get_default_config()
        assert flow_search.vector_store is None
        assert flow_search._model_storage == model_storage
        assert flow_search.flow_document_template is not None

    def test_generate_flow_documents(
        self,
        flow_search: FlowRetrieval,
        flows: FlowsList,
        domain: Mock,
        startable_flows_documents: List[Document],
    ):
        def get_flow_id(d: Document):
            return d.metadata["flow_id"]

        """Test that generate_flow_documents renders the correct template string."""
        # Given
        flow_search.config[SHOULD_EMBED_SLOTS_KEY] = True
        # our flow document fixture doesn't contain the document
        # for link test flow
        startable_flows = FlowsList(
            [f for f in flows if not f.is_startable_only_via_link()]
        )
        # When
        docs = flow_search._generate_flow_documents(startable_flows, domain)
        # Then
        assert len(docs) == 2
        assert sorted(docs, key=get_flow_id) == sorted(
            startable_flows_documents, key=get_flow_id
        )

    @pytest.mark.parametrize(
        "config",
        [
            {
                EMBEDDINGS_CONFIG_KEY: {
                    "api_type": "openai",
                    "model": "some_custom_option",
                }
            },
            FlowRetrieval.get_default_config(),
        ],
    )
    @patch("langchain.vectorstores.faiss.FAISS.from_documents")
    def test_populate(
        self,
        mock_faiss_from_documents: Mock,
        config: Dict,
        model_storage: ModelStorage,
        resource: Resource,
        flows: FlowsList,
        domain: Mock,
        startable_flows_documents,
        monkeypatch: MonkeyPatch,
    ) -> None:
        # Given
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        mock_faiss_from_documents.return_value = Mock()
        flow_search = FlowRetrieval(config, model_storage, resource)

        # When
        flow_search.populate(flows, domain)
        # Then
        # even if we passed all flows, we are expecting that the
        # FlowSearch._generate_documents is only going to return
        # the documents from flows that are not link-accessed only
        assert len(mock_faiss_from_documents.call_args.kwargs) == 3

        embedder = mock_faiss_from_documents.call_args.kwargs["embedding"]
        assert isinstance(embedder, _LangchainEmbeddingClientAdapter)
        assert isinstance(embedder._client, OpenAIEmbeddingClient)
        assert embedder._client.model == config[EMBEDDINGS_CONFIG_KEY]["model"]

        assert (
            mock_faiss_from_documents.call_args.kwargs["documents"]
            == startable_flows_documents
        )
        assert (
            mock_faiss_from_documents.call_args.kwargs["distance_strategy"]
            == DistanceStrategy.MAX_INNER_PRODUCT
        )

    @pytest.mark.parametrize(
        "max_flows_from_semantic_search, expected_flows_from_vector_store",
        [
            (-1, []),
            (1, ["test_always_included_flow"]),
            (2, ["test_always_included_flow", "test_flow_with_collect_steps"]),
        ],
    )
    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval."
        "FlowRetrieval.find_most_similar_flows"
    )
    async def test_filter_flows(
        self,
        mock_find_most_similar_flows: Mock,
        max_flows_from_semantic_search: int,
        expected_flows_from_vector_store: List[Text],
        flow_search: FlowRetrieval,
        flows: FlowsList,
    ):
        # Given
        flow_search.vector_store = Mock()
        flow_search.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY] = (
            max_flows_from_semantic_search
        )
        # return the flows expected from similarity search
        mock_find_most_similar_flows.return_value = FlowsList(
            [f for f in flows if f.id in expected_flows_from_vector_store]
        )
        # merge flows returned by similarity search and by always include flag
        # to create full list of expected flows
        expected_flows = FlowsList.from_multiple_flows_lists(
            mock_find_most_similar_flows.return_value,
            FlowsList([flows.flow_by_id("test_always_included_flow")]),
        )
        # tracker that contains flows defined in the 'flows' fixture
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[
                UserUttered("Hello"),
                BotUttered("Hi"),
                UserUttered("Start flow always included in prompt"),
                # this flow should be included
                FlowStarted(flow_id="test_always_included_flow"),
                FlowCompleted(
                    flow_id="test_always_included_flow", step_id="first_step_a"
                ),
                # this flow should not be included
                FlowStarted(flow_id="test_link_based_flow"),
                FlowCompleted(flow_id="test_link_based_flow", step_id="first_step_b"),
            ],
        )
        # When
        filtered_flows = await flow_search.filter_flows(tracker, Mock(), flows)
        # Then
        assert len(filtered_flows) == len(expected_flows)
        assert filtered_flows.user_flow_ids == expected_flows.user_flow_ids

    @pytest.mark.parametrize(
        "conversation_turns_to_embed, expected_query",
        (
            # last 10 messages
            (
                10,
                (
                    f"{USER}: Hello\n"
                    f"{AI}: Hello! How can I help you?\n"
                    f"{USER}: Transfer $10 to John"
                ),
            ),
            # only last message
            (0, "Transfer $10 to John"),
        ),
    )
    def test_prepare_query(
        self,
        conversation_turns_to_embed: int,
        expected_query: Text,
        flow_search: FlowRetrieval,
    ):
        # Given
        flow_search.config[TURNS_TO_EMBED_KEY] = conversation_turns_to_embed
        tracker = DialogueStateTracker.from_events(
            sender_id="test",
            evts=[
                UserUttered("Hello"),
                BotUttered("Hello! How can I help you?"),
            ],
        )
        user_message = Message(data={TEXT: "Transfer $10 to John"})
        # When
        query = flow_search._prepare_query(tracker, user_message)
        # Then
        assert query == expected_query

    @patch("langchain.vectorstores.faiss.FAISS.asimilarity_search_with_score")
    async def test_query_vector_store(
        self,
        mock_asimilarity_search_with_score: Mock,
        flow_search: FlowRetrieval,
    ):
        # Given
        query = "test query"
        flow_search.vector_store = FAISS(Mock(), Mock(), Mock(), Mock())
        k = flow_search.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY]
        # When
        await flow_search._query_vector_store(query)
        # Then
        mock_asimilarity_search_with_score.assert_called_once_with(query, k=k)

    @patch("langchain.vectorstores.faiss.FAISS.asimilarity_search_with_score")
    async def test_query_vector_store_throws_exception(
        self,
        mock_asimilarity_search_with_score: Mock,
        flow_search: FlowRetrieval,
    ):
        # Given
        query = "test query"
        flow_search.vector_store = FAISS(Mock(), Mock(), Mock(), Mock())
        k = flow_search.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY]
        mock_asimilarity_search_with_score.side_effect = Exception("Test Exception")
        # When
        with pytest.raises(Exception) as exc_info:
            await flow_search._query_vector_store(query)
        # Then
        assert "Test Exception" in str(exc_info.value), "Expected exception not raised"
        mock_asimilarity_search_with_score.assert_called_once_with(query, k=k)

    async def test_query_vector_store_when_its_not_initialized(
        self,
        flow_search: FlowRetrieval,
    ):
        # Given
        query = "test query"
        # When
        result = await flow_search._query_vector_store(query)
        # Then
        assert result == []

    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._query_vector_store"
    )
    @patch(
        "rasa.dialogue_understanding.generator.flow_retrieval.FlowRetrieval._prepare_query"
    )
    async def test_find_most_similar_flows(
        self,
        mock_prepare_query: Mock,
        mock_query_vector_store: Mock,
        flows: FlowsList,
        startable_flows_documents: List[Document],
        flow_search: FlowRetrieval,
    ):
        # Given
        flow_search.vector_store = FAISS(Mock(), Mock(), Mock(), Mock())
        query = "test user message"
        mock_prepare_query.return_value = query
        mock_query_vector_store.return_value = [
            (d, 1.0) for d in startable_flows_documents
        ]
        # When
        most_similar_flows = await flow_search.find_most_similar_flows(
            tracker=Mock(), message=Mock(), flows=flows
        )
        # Then
        mock_prepare_query.assert_called_once()
        mock_query_vector_store.assert_called_once_with(query)
        assert len(most_similar_flows) == 2
        assert most_similar_flows.user_flow_ids == {
            "test_always_included_flow",
            "test_flow_with_collect_steps",
        }

    @patch("langchain.vectorstores.faiss.FAISS.save_local")
    def persist_vector_store(
        self,
        mock_faiss_save_local: Mock,
        flow_search: FlowRetrieval,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # Given
        flow_search.vector_store = FAISS(Mock(), Mock(), Mock(), Mock())
        # When
        flow_search._persist_vector_store()
        # Then
        mock_faiss_save_local.assert_called_once()

    @patch("langchain.vectorstores.faiss.FAISS.save_local")
    def persist_vector_store_not_initialized(
        self,
        mock_faiss_save_local: Mock,
        flow_search: FlowRetrieval,
        model_storage: ModelStorage,
        resource: Resource,
    ):
        # Given
        flow_search.vector_store = None
        # When
        flow_search._persist_vector_store()
        # Then
        mock_faiss_save_local.assert_not_called()

    @patch("langchain.vectorstores.faiss.FAISS.save_local")
    @patch("langchain.vectorstores.faiss.FAISS.load_local")
    def test_load_vector_store(
        self,
        mock_load_local: Mock,
        mock_save_local: Mock,
        flow_search: FlowRetrieval,
        model_storage: ModelStorage,
        resource: Resource,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Given
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        expected_vector_store = FAISS(Mock(), Mock(), Mock(), Mock())
        mock_load_local.return_value = expected_vector_store
        config = FlowRetrieval.get_default_config()
        flow_search.vector_store = expected_vector_store
        flow_search.persist()
        # When
        flow_search._load_vector_store(config, model_storage, resource)
        # Then
        mock_load_local.assert_called_once()
        assert flow_search.vector_store == expected_vector_store

    def test_load_vector_store_failure(
        self,
        flow_search: FlowRetrieval,
        model_storage: ModelStorage,
        resource: Resource,
        monkeypatch: MonkeyPatch,
    ):
        # Given
        config = FlowRetrieval.get_default_config()
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        # there is no vector_store initialized
        flow_search.persist()
        # When
        flow_search._load_vector_store(config, model_storage, resource)
        # Then
        assert flow_search.vector_store is None

    @pytest.mark.parametrize(
        "key, value, expected_value",
        (
            (TURNS_TO_EMBED_KEY, -1, DEFAULT_TURNS_TO_EMBED),
            (TURNS_TO_EMBED_KEY, 0, DEFAULT_TURNS_TO_EMBED),
            (TURNS_TO_EMBED_KEY, 10, 10),
            (TURNS_TO_EMBED_KEY, 1, 1),
            (
                MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY,
                -1,
                DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH,
            ),
            (MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY, 0, 0),
            (MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY, 10, 10),
        ),
    )
    def test_validate_config(self, key: Text, value: Any, expected_value: Any):
        # Given
        config = FlowRetrieval.get_default_config()
        config[key] = value
        # When
        FlowRetrieval.validate_config(config)
        # Then
        assert config[key] == expected_value
