"""The module is primarily centered around the `FlowRetrieval` class which handles the
initialization, configuration validation, vector store management, and flow retrieval
logic. It integrates components for managing embeddings, vector stores, and
flow-specific templates, facilitating semantic search functionalities.
Key Features:
- Configurable embedding strategies for dialogue components.
- Seamless interaction with model storage and resource management.
- Supports dynamic loading and persistence of vector stores.
- Enables population of vector stores based on specified dialogue flows and domain
information.
- Implements flow retrieval mechanisms including semantic search based on dialogue
context.
Usage:
Interaction with this class typically involves creating an instance with a
configuration dict, model storage, and a resource reference, then using methods
like `populate`, `persist`, or dynamic retrieval methods to manage or utilize
flows within a conversational context.
"""

import importlib
from typing import Any, Dict, List, Optional, Text

import structlog
from jinja2 import Template
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

import rasa.shared.utils.io
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROVIDER_CONFIG_KEY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.nlu.constants import FLOWS_FROM_SEMANTIC_SEARCH, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.providers.embedding._langchain_embedding_client_adapter import (
    _LangchainEmbeddingClientAdapter,
    embedding_metadata_context,
)
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
)
from rasa.shared.utils.health_check.embeddings_health_check_mixin import (
    EmbeddingsHealthCheckMixin,
)
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    USER,
    allowed_values_for_slot,
    embedder_factory,
    resolve_model_client_config,
    tracker_as_readable_transcript,
)

DEFAULT_FLOW_DOCUMENT_TEMPLATE = importlib.resources.read_text(
    "rasa.dialogue_understanding.generator", "flow_document_template.jinja2"
)

FLOW_RETRIEVAL_CONFIG_FILE_NAME = "flow_retrieval_config.json"

DEFAULT_EMBEDDINGS_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    "model": DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY = "num_flows"
SHOULD_EMBED_SLOTS_KEY = "should_embed_slots"
TURNS_TO_EMBED_KEY = "turns_to_embed"

DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH = 20
DEFAULT_TURNS_TO_EMBED = 1
DEFAULT_SHOULD_EMBED_SLOTS = True


structlogger = structlog.get_logger()


class FlowRetrieval(EmbeddingsHealthCheckMixin):
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """The default config for the flow retrieval."""
        return {
            EMBEDDINGS_CONFIG_KEY: DEFAULT_EMBEDDINGS_CONFIG,
            MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY: DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH,
            TURNS_TO_EMBED_KEY: DEFAULT_TURNS_TO_EMBED,
            SHOULD_EMBED_SLOTS_KEY: DEFAULT_SHOULD_EMBED_SLOTS,
        }

    def __init__(
        self,
        config: Dict[str, Any],
        model_storage: ModelStorage,
        resource: Resource,
    ):
        config = {**self.get_default_config(), **config}
        self.config = self.validate_config(config)
        self.config[EMBEDDINGS_CONFIG_KEY] = resolve_model_client_config(
            self.config.get(EMBEDDINGS_CONFIG_KEY), FlowRetrieval.__name__
        )
        self.vector_store: Optional[FAISS] = None
        self.flow_document_template = DEFAULT_FLOW_DOCUMENT_TEMPLATE
        self._model_storage = model_storage
        self._resource = resource

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> Dict[Text, Any]:
        if config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY] < 0:
            config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY] = (
                DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH
            )
            structlogger.error(
                f"flow_retrieval.validate_config.{MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY}.set_as_negative",
                event_info=(
                    f"The `{MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY}` is a "
                    f"negative value. Setting it to the default value "
                    f"({DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH})."
                ),
                old_value=config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY],
                new_value=DEFAULT_MAX_FLOWS_FROM_SEMANTIC_SEARCH,
            )

        if config[TURNS_TO_EMBED_KEY] < 1:
            config[TURNS_TO_EMBED_KEY] = DEFAULT_TURNS_TO_EMBED
            structlogger.error(
                f"flow_retrieval.validate_config.{TURNS_TO_EMBED_KEY}.less_than_one",
                event_info=(
                    f"The `{TURNS_TO_EMBED_KEY}` is less than 1."
                    f"Setting it to the default value ({DEFAULT_TURNS_TO_EMBED})."
                    f"Only the latest user utterance will be used."
                ),
                old_value=config[TURNS_TO_EMBED_KEY],
                new_value=DEFAULT_TURNS_TO_EMBED,
            )

        return config

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        **kwargs: Any,
    ) -> "FlowRetrieval":
        """Load flow retrieval with previously populated FAISS vector store."""
        # Perform health check on resolved embedding client config
        embeddings_config = resolve_model_client_config(
            config.get(EMBEDDINGS_CONFIG_KEY, {})
        )
        cls.perform_embeddings_health_check(
            embeddings_config,
            DEFAULT_EMBEDDINGS_CONFIG,
            "flow_retrieval.load",
            FlowRetrieval.__name__,
        )

        # initialize base flow retrieval
        flow_retrieval = FlowRetrieval(config, model_storage, resource)
        # load vector store
        vector_store = cls._load_vector_store(
            flow_retrieval.config, model_storage, resource
        )
        flow_retrieval.vector_store = vector_store

        return flow_retrieval

    @classmethod
    def _load_vector_store(
        cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource
    ) -> Optional[FAISS]:
        """Loads a FAISS vector store from a specified local path."""
        embeddings = cls._create_embedder(config)
        try:
            with model_storage.read_from(resource) as model_path:
                return FAISS.load_local(
                    folder_path=model_path,
                    embeddings=embeddings,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
                    allow_dangerous_deserialization=True,
                )
        except Exception as e:
            structlogger.warning(
                "flow_retrieval.load_vector_store.failed",
                error=e,
                resource=resource.name,
            )
            return None

    @classmethod
    def _create_embedder(cls, config: Dict[Text, Any]) -> Embeddings:
        """Creates an embedder.

        Returns:
            The embedder.
        """
        # Copy the config so original config is not modified
        config = config.copy()
        # Resolve config and instantiate the embedding client
        config[EMBEDDINGS_CONFIG_KEY] = resolve_model_client_config(
            config.get(EMBEDDINGS_CONFIG_KEY), FlowRetrieval.__name__
        )
        client = embedder_factory(
            config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )
        # Wrap the embedding client in the adapter
        return _LangchainEmbeddingClientAdapter(client)

    def persist(self) -> None:
        self._persist_vector_store()
        self._persist_config()

    def _persist_vector_store(self) -> None:
        """Persists the FAISS vector store."""
        if self.vector_store is not None:
            with self._model_storage.write_to(self._resource) as model_path:
                self.vector_store.save_local(model_path)

    def _persist_config(self) -> None:
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                path / FLOW_RETRIEVAL_CONFIG_FILE_NAME, self.config
            )

    def get_llm_tracing_metadata(self, tracker: DialogueStateTracker) -> Dict[str, Any]:
        return {
            LANGFUSE_METADATA_SESSION_ID: tracker.sender_id,
            LANGFUSE_METADATA_TAGS: [self.__class__.__name__],
            LANGFUSE_METADATA_CUSTOM_METADATA: {
                LANGFUSE_METADATA_AGENT_ID: tracker.assistant_id,
                LANGFUSE_METADATA_MODEL_ID: tracker.model_id,
                LANGFUSE_METADATA_COMPONENT_NAME: self.__class__.__name__,
            },
        }

    def populate(self, flows: FlowsList, domain: Domain) -> None:
        """Populates the vector store with embeddings generated from
        documents based on the flow descriptions, and flow slots
        descriptions.

        Args:
            flows: List of flows to populate the vector store with.
            domain: The domain containing relevant slot information.
        """
        # Perform health check before populating the vector store with flows
        self.perform_embeddings_health_check(
            self.config.get(EMBEDDINGS_CONFIG_KEY),
            DEFAULT_EMBEDDINGS_CONFIG,
            "flow_retrieval.train",
            FlowRetrieval.__name__,
        )

        flows_to_embedd = flows.exclude_link_only_flows()

        if not flows_to_embedd:
            structlogger.debug(
                "flow_retrieval.populate_vector_store.no_flows_to_embed",
                event_info=(
                    "No flows to embed in the vector store, skipping population."
                ),
            )
            return

        embeddings = self._create_embedder(self.config)
        documents = self._generate_flow_documents(flows_to_embedd, domain)
        try:
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )
        except Exception as e:
            error_type = e.__class__.__name__
            structlogger.error(
                "flow_retrieval.populate_vector_store.not_populated",
                event_info=(
                    "Failed to populate the FAISS store with the provided flows."
                ),
                error_type=error_type,
                error=e,
            )
            raise

    def _generate_flow_documents(
        self, flows: FlowsList, domain: Domain
    ) -> List[Document]:
        """Converts each flow in the provided list into an embeddable document. These
        documents include the following information for each flow: flow's name,
        flow's description, and associated slots and their descriptions and allowed
        values.

        Args:
            flows: List of flows.
            domain: The domain containing relevant slot information.

        Returns:
            List of documents, each representing the converted information of a single
            flow.
        """
        slots = {slot.name: slot for slot in domain.slots}
        flow_docs = []

        for flow in flows:
            flow_info: Dict[Text, Any] = {
                "flow": {
                    "name": flow.name,
                    "description": flow.description,
                    "slots": [],
                }
            }
            if self.config[SHOULD_EMBED_SLOTS_KEY]:
                flow_info["flow"]["slots"] = [
                    {
                        "name": q.collect,
                        "description": q.description,
                        "allowed_values": allowed_values_for_slot(slots[q.collect]),
                    }
                    for q in flow.get_collect_steps()
                ]

            flow_docs.append(
                Document(
                    page_content=Template(self.flow_document_template)
                    .render(flow_info)
                    .strip(),
                    metadata={"flow_id": flow.id},
                )
            )

        return flow_docs

    async def filter_flows(
        self, tracker: DialogueStateTracker, message: Message, flows: FlowsList
    ) -> FlowsList:
        """Filters the given flows.
        The filtered flows with the following rules:
        - Exclude flows that reachable only via link
        - Include flows set to be always included in the prompt
        - Include flows started during the conversation,
        - (Optionally) The rest of flows are filter so only the top 'k' with
          the highest similarity to the ongoing conversation are included.
          Set 'max_flows_from_semantic_search' to negative number to disable
          the option.

        Args:
            tracker: the tracker
            message: the user message
            flows: available flows

        Returns:
            List of flows that includes always-included flows, previously started flows
            and top `k` relevant flows for the current conversation.
        """
        # apply basic filtering
        flows = flows.exclude_link_only_flows()
        always_included_flows = flows.get_flows_always_included_in_prompt()
        previously_started_flows = tracker.get_previously_started_flows(flows)
        # apply semantic search filtering
        most_similar_flows = await self.find_most_similar_flows(tracker, message, flows)
        return FlowsList.from_multiple_flows_lists(
            always_included_flows,
            previously_started_flows,
            most_similar_flows,
        )

    async def find_most_similar_flows(
        self, tracker: DialogueStateTracker, message: Message, flows: FlowsList
    ) -> FlowsList:
        """Filters the given flows so only the top 'k' most similar
        flows to the current conversation are left.
        retrieved from the vector store.

        Args:
            tracker: the tracker
            message: the user message

        Returns:
            The most similar flows to the current conversation.
        """
        query = self._prepare_query(tracker, message)
        documents_with_scores = await self._query_vector_store(query, tracker)
        # filter out None i.e. more flows were embedded during training than are
        # available during prediction
        most_similar_flows_with_scores = [
            (flow, score)
            for doc, score in documents_with_scores
            if (flow := flows.flow_by_id(doc.metadata["flow_id"])) is not None
        ]
        # sort by score decreasing
        most_similar_flows_with_scores.sort(key=lambda x: x[1], reverse=True)
        # add the relevant flows to the message for evaluation purposes
        message.set(
            prop=FLOWS_FROM_SEMANTIC_SEARCH,
            info=[
                (flow.id, float(score))
                for flow, score in most_similar_flows_with_scores
            ],
            add_to_output=True,
        )
        return FlowsList([f for f, _ in most_similar_flows_with_scores])

    def _prepare_query(self, tracker: DialogueStateTracker, message: Message) -> str:
        """Prepares the query for vector store. The query is composed
        of the conversation within the specified number of turns.

        Args:
            tracker: the tracker
            message: the user message

        Returns:
            The query for vector store.
        """
        if int(self.config[TURNS_TO_EMBED_KEY]) > 1:
            current_conversation = tracker_as_readable_transcript(
                tracker,
                human_prefix=USER,
                max_turns=int(self.config[TURNS_TO_EMBED_KEY]),
            )
            current_conversation += f"\n{USER}: {message.data[TEXT]}"
            return current_conversation

        return f"{message.data[TEXT]}"

    async def _query_vector_store(
        self, query: str, tracker: DialogueStateTracker
    ) -> List:
        """Compares the query with all flows using a vector store
        and returns the top k relevant flows for the current conversation.

        Args:
            query: query

        Returns:
            The top k documents with similarity scores.
        """
        if self.vector_store is None:
            return []
        try:
            with embedding_metadata_context(self.get_llm_tracing_metadata(tracker)):
                documents_with_scores = (
                    await self.vector_store.asimilarity_search_with_score(
                        query, k=int(self.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY])
                    )
                )
            structlogger.debug(
                "flow_retrieval.query_vector_store.fetched",
                event_info=(
                    f"Fetched the top {self.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY]}"
                    f"similar flows from the vector store"
                ),
                query=query,
                top_k=self.config[MAX_FLOWS_FROM_SEMANTIC_SEARCH_KEY],
                results=[
                    {
                        "flow_id": document.metadata["flow_id"],
                        "score": score,
                        "content": document.page_content,
                    }
                    for document, score in documents_with_scores
                ],
            )
            return documents_with_scores
        except Exception as e:
            error_type = e.__class__.__name__
            structlogger.error(
                "flow_retrieval.query_vector_store.error",
                event_info="Cannot fetch flows from vector store",
                error_type=error_type,
                error=e,
                query=query,
            )
            raise ProviderClientAPIException(
                message="Cannot fetch flows from vector store", original_exception=e
            )
