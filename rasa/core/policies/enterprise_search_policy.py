import importlib.resources
import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

import dotenv
import structlog
from jinja2 import Template
from pydantic import ValidationError

from rasa.shared.providers.embedding._langchain_embedding_client_adapter import (
    _LangchainEmbeddingClientAdapter,
)
from rasa.shared.providers.llm.llm_client import LLMClient

import rasa.shared.utils.io
from rasa.telemetry import (
    track_enterprise_search_policy_predict,
    track_enterprise_search_policy_train_completed,
    track_enterprise_search_policy_train_started,
)
from rasa.shared.exceptions import RasaException
from rasa.core.constants import (
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
    SEARCH_POLICY_PRIORITY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.utils import AvailableEndpoints
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.dialogue_understanding.stack.frames import (
    DialogueStackFrame,
    SearchStackFrame,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.forms_provider import Forms
from rasa.graph_components.providers.responses_provider import Responses
from rasa.shared.constants import (
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    PROMPT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    OPENAI_PROVIDER,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.constants import (
    ACTION_CANCEL_FLOW,
    ACTION_SEND_TEXT_NAME,
    DEFAULT_SLOT_NAMES,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, UserUttered, BotUttered
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker, EventVerbosity
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.cli import print_error_and_exit
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    embedder_factory,
    get_prompt_template,
    llm_factory,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
    try_instantiate_llm_client,
)
from rasa.core.information_retrieval.faiss import FAISS_Store
from rasa.core.information_retrieval import (
    InformationRetrieval,
    SearchResult,
    InformationRetrievalException,
    create_from_endpoint_config,
)

if TYPE_CHECKING:
    from langchain.schema.embeddings import Embeddings
    from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer

from rasa.utils.log_utils import log_llm

logger = structlog.get_logger()

dotenv.load_dotenv("./.env")

SOURCE_PROPERTY = "source"
VECTOR_STORE_TYPE_PROPERTY = "type"
VECTOR_STORE_PROPERTY = "vector_store"
VECTOR_STORE_THRESHOLD_PROPERTY = "threshold"
TRACE_TOKENS_PROPERTY = "trace_prompt_tokens"
CITATION_ENABLED_PROPERTY = "citation_enabled"
USE_LLM_PROPERTY = "use_generative_llm"
MAX_MESSAGES_IN_QUERY_KEY = "max_messages_in_query"

DEFAULT_VECTOR_STORE_TYPE = "faiss"
DEFAULT_VECTOR_STORE_THRESHOLD = 0.0
DEFAULT_VECTOR_STORE = {
    VECTOR_STORE_TYPE_PROPERTY: DEFAULT_VECTOR_STORE_TYPE,
    SOURCE_PROPERTY: "./docs",
    VECTOR_STORE_THRESHOLD_PROPERTY: DEFAULT_VECTOR_STORE_THRESHOLD,
}

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_CHAT_MODEL_NAME,
    TIMEOUT_CONFIG_KEY: 10,
    "temperature": 0.0,
    "max_tokens": 256,
    "max_retries": 1,
}

DEFAULT_EMBEDDINGS_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    "model": DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

ENTERPRISE_SEARCH_PROMPT_FILE_NAME = "enterprise_search_policy_prompt.jinja2"

SEARCH_RESULTS_METADATA_KEY = "search_results"
SEARCH_QUERY_METADATA_KEY = "search_query"

DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.core.policies", "enterprise_search_prompt_template.jinja2"
)

DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE = importlib.resources.read_text(
    "rasa.core.policies", "enterprise_search_prompt_with_citation_template.jinja2"
)


class VectorStoreConnectionError(RasaException):
    """Exception raised for errors in connecting to the vector store."""


class VectorStoreConfigurationError(RasaException):
    """Exception raised for errors in vector store configuration."""


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class EnterpriseSearchPolicy(Policy):
    """Policy which uses a vector store and LLMs to respond to user messages.

    The policy uses a vector store and LLMs to respond to user messages. The
    vector store is used to retrieve the most relevant responses to the user
    message. The LLMs are used to rank the responses and select the best
    response. The policy can be used to respond to user messages without
    training data.

    Example Configuration:

        policies:
            # - ...
            - name: EnterpriseSearchPolicy
              vector_store:
                type: "milvus"
                <vector_store_config>
            # - ...
    """

    @staticmethod
    def does_support_stack_frame(frame: DialogueStackFrame) -> bool:
        """Checks if the policy supports the given stack frame."""
        return isinstance(frame, SearchStackFrame)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Returns the default config of the policy."""
        return {
            POLICY_PRIORITY: SEARCH_POLICY_PRIORITY,
            VECTOR_STORE_PROPERTY: DEFAULT_VECTOR_STORE,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        vector_store: Optional[InformationRetrieval] = None,
        featurizer: Optional["TrackerFeaturizer"] = None,
        prompt_template: Optional[Text] = None,
    ) -> None:
        """Constructs a new Policy object."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)

        # Vector store object and configuration
        self.vector_store = vector_store
        self.vector_store_config = config.get(
            VECTOR_STORE_PROPERTY, DEFAULT_VECTOR_STORE
        )
        # Embeddings configuration for encoding the search query
        self.embeddings_config = self.config.get(
            EMBEDDINGS_CONFIG_KEY, DEFAULT_EMBEDDINGS_CONFIG
        )
        # Maximum number of turns to include in the prompt
        self.max_history = self.config.get(POLICY_MAX_HISTORY)

        # Maximum number of messages to include in the search query
        self.max_messages_in_query = self.config.get(MAX_MESSAGES_IN_QUERY_KEY, 2)

        # LLM Configuration for response generation
        self.llm_config = self.config.get(LLM_CONFIG_KEY, DEFAULT_LLM_CONFIG)

        # boolean to enable/disable tracing of prompt tokens
        self.trace_prompt_tokens = self.config.get(TRACE_TOKENS_PROPERTY, False)

        # boolean to enable/disable the use of LLM for response generation
        self.use_llm = self.config.get(USE_LLM_PROPERTY, True)

        # boolean to enable/disable citation generation
        self.citation_enabled = self.config.get(CITATION_ENABLED_PROPERTY, False)

        self.prompt_template = prompt_template or get_prompt_template(
            self.config.get(PROMPT_CONFIG_KEY),
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
        )
        self.citation_prompt_template = get_prompt_template(
            self.config.get(PROMPT_CONFIG_KEY),
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_WITH_CITATION_TEMPLATE,
        )
        # If citation is enabled, use the citation prompt template
        if self.citation_enabled:
            self.prompt_template = self.citation_prompt_template

    @classmethod
    def _create_plain_embedder(cls, config: Dict[Text, Any]) -> "Embeddings":
        """Creates an embedder based on the given configuration.

        Returns:
        The embedder.
        """
        client = embedder_factory(
            config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )
        return _LangchainEmbeddingClientAdapter(client)

    def train(  # type: ignore[override]
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        responses: Responses,
        forms: Forms,
        training_data: TrainingData,
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            responses: The model's responses.
            forms: The model's forms.
            training_data: The model's training data.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        store_type = self.vector_store_config.get(VECTOR_STORE_TYPE_PROPERTY)

        # telemetry call to track training start
        track_enterprise_search_policy_train_started()

        # validate embedding configuration
        try:
            embeddings = self._create_plain_embedder(self.config)
        except (ValidationError, Exception) as e:
            logger.error(
                "enterprise_search_policy.train.embedder_instantiation_failed",
                message="Unable to instantiate the embedding client.",
                error=e,
            )
            print_error_and_exit(
                "Unable to create embedder. Please make sure you specified the "
                f"required environment variables. Error: {e}"
            )

        # validate llm configuration
        try_instantiate_llm_client(
            self.config.get(LLM_CONFIG_KEY),
            DEFAULT_LLM_CONFIG,
            "enterprise_search_policy.train",
            "EnterpriseSearchPolicy",
        )

        if store_type == DEFAULT_VECTOR_STORE_TYPE:
            logger.info("enterprise_search_policy.train.faiss")
            with self._model_storage.write_to(self._resource) as path:
                self.vector_store = FAISS_Store(
                    docs_folder=self.vector_store_config.get(SOURCE_PROPERTY),
                    embeddings=embeddings,
                    index_path=path,
                    create_index=True,
                )
        else:
            logger.info("enterprise_search_policy.train.custom", store_type=store_type)

        # telemetry call to track training completion
        track_enterprise_search_policy_train_completed(
            vector_store_type=store_type,
            embeddings_type=self.embeddings_config.get(PROVIDER_CONFIG_KEY),
            embeddings_model=self.embeddings_config.get(MODEL_CONFIG_KEY)
            or self.embeddings_config.get(MODEL_NAME_CONFIG_KEY),
            llm_type=self.llm_config.get(PROVIDER_CONFIG_KEY),
            llm_model=self.llm_config.get(MODEL_CONFIG_KEY)
            or self.llm_config.get(MODEL_NAME_CONFIG_KEY),
            citation_enabled=self.citation_enabled,
        )
        self.persist()
        return self._resource

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / ENTERPRISE_SEARCH_PROMPT_FILE_NAME
            )

    def _prepare_slots_for_template(
        self, tracker: DialogueStateTracker
    ) -> List[Dict[str, str]]:
        """Prepares the slots for the template.

        Args:
            tracker: The tracker containing the conversation history up to now.

        Returns:
            The non-empty slots.
        """
        template_slots = []
        for name, slot in tracker.slots.items():
            if name not in DEFAULT_SLOT_NAMES and slot.value is not None:
                template_slots.append(
                    {
                        "name": name,
                        "value": str(slot.value),
                        "type": slot.type_name,
                    }
                )
        return template_slots

    def _connect_vector_store_or_raise(
        self, endpoints: Optional[AvailableEndpoints]
    ) -> None:
        """Connects to the vector store or raises an exception.

        Raise exceptions for the following cases:
        - The configuration is not specified
        - Unable to connect to the vector store

        Args:
            endpoints: Endpoints configuration.
        """
        config = endpoints.vector_store if endpoints else None
        store_type = self.vector_store_config.get(VECTOR_STORE_TYPE_PROPERTY)
        if config is None and store_type != DEFAULT_VECTOR_STORE_TYPE:
            logger.error(
                "enterprise_search_policy._connect_vector_store_or_raise.no_config"
            )
            raise VectorStoreConfigurationError(
                """No vector store specified. Please specify a vector
                store in the endpoints configuration"""
            )
        try:
            self.vector_store.connect(config)  # type: ignore
        except Exception as e:
            logger.error(
                "enterprise_search_policy._connect_vector_store_or_raise.connect_error",
                error=e,
                config=config,
            )
            raise VectorStoreConnectionError(
                f"Unable to connect to the vector store. Error: {e}"
            )

    def _prepare_search_query(self, tracker: DialogueStateTracker, history: int) -> str:
        """Prepares the search query.
        The search query is the last N messages in the conversation history.

        Args:
            tracker: The tracker containing the conversation history up to now.
            history: The number of messages to include in the search query.

        Returns:
            The search query.
        """
        transcript = []
        for event in tracker.applied_events():
            if isinstance(event, UserUttered) or isinstance(event, BotUttered):
                transcript.append(sanitize_message_for_prompt(event.text))

        search_query = " ".join(transcript[-history:][::-1])
        logger.debug("search_query", search_query=search_query)
        return search_query

    async def predict_action_probabilities(  # type: ignore[override]
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        endpoints: Optional[AvailableEndpoints],
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            endpoints: The model's endpoints.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        logger_key = "enterprise_search_policy.predict_action_probabilities"
        vector_search_threshold = self.vector_store_config.get(
            VECTOR_STORE_THRESHOLD_PROPERTY, DEFAULT_VECTOR_STORE_THRESHOLD
        )
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)
        if not self.supports_current_stack_frame(
            tracker, False, False
        ) or self.should_abstain_in_coexistence(tracker, True):
            return self._prediction(self._default_predictions(domain))

        if not self.vector_store:
            logger.error(f"{logger_key}.no_vector_store")
            return self._create_prediction_internal_error(domain, tracker)

        try:
            self._connect_vector_store_or_raise(endpoints)
        except (VectorStoreConfigurationError, VectorStoreConnectionError) as e:
            logger.error(f"{logger_key}.connection_error", error=e)
            return self._create_prediction_internal_error(domain, tracker)

        search_query = self._prepare_search_query(
            tracker, int(self.max_messages_in_query)
        )
        tracker_state = tracker.current_state(EventVerbosity.AFTER_RESTART)

        try:
            documents = await self.vector_store.search(
                query=search_query,
                tracker_state=tracker_state,
                threshold=vector_search_threshold,
            )
        except InformationRetrievalException as e:
            logger.error(f"{logger_key}.search_error", error=e)
            return self._create_prediction_internal_error(domain, tracker)

        if not documents.results:
            logger.info(f"{logger_key}.no_documents")
            return self._create_prediction_cannot_handle(domain, tracker)

        if self.use_llm:
            prompt = self._render_prompt(tracker, documents.results)
            llm_answer = await self._generate_llm_answer(llm, prompt)

            if self.citation_enabled:
                llm_answer = self.post_process_citations(llm_answer)

            logger.debug(f"{logger_key}.llm_answer", llm_answer=llm_answer)
            response = llm_answer
        else:
            response = documents.results[0].metadata.get("answer", None)
            if not response:
                logger.error(
                    f"{logger_key}.answer_key_missing_in_metadata",
                    search_results=documents.results,
                )
            logger.debug(
                "enterprise_search_policy.predict_action_probabilities.no_llm",
                search_results=documents,
            )

        if response is None:
            return self._create_prediction_internal_error(domain, tracker)

        action_metadata = {
            "message": {
                "text": response,
                SEARCH_RESULTS_METADATA_KEY: [
                    result.text for result in documents.results
                ],
                UTTER_SOURCE_METADATA_KEY: self.__class__.__name__,
                SEARCH_QUERY_METADATA_KEY: search_query,
            }
        }

        # telemetry call to track policy prediction
        track_enterprise_search_policy_predict(
            vector_store_type=self.vector_store_config.get(VECTOR_STORE_TYPE_PROPERTY),
            embeddings_type=self.embeddings_config.get(PROVIDER_CONFIG_KEY),
            embeddings_model=self.embeddings_config.get(MODEL_CONFIG_KEY)
            or self.embeddings_config.get(MODEL_NAME_CONFIG_KEY),
            llm_type=self.llm_config.get(PROVIDER_CONFIG_KEY),
            llm_model=self.llm_config.get(MODEL_CONFIG_KEY)
            or self.llm_config.get(MODEL_NAME_CONFIG_KEY),
            citation_enabled=self.citation_enabled,
        )
        return self._create_prediction(
            domain=domain, tracker=tracker, action_metadata=action_metadata
        )

    def _render_prompt(
        self, tracker: DialogueStateTracker, documents: List[SearchResult]
    ) -> Text:
        """Renders the prompt from the template.

        Args:
            tracker: The tracker containing the conversation history up to now.
            documents: The documents retrieved from search

        Returns:
            The rendered prompt.
        """
        inputs = {
            "current_conversation": tracker_as_readable_transcript(
                tracker, max_turns=self.max_history
            ),
            "docs": documents,
            "slots": self._prepare_slots_for_template(tracker),
            "citation_enabled": self.citation_enabled,
        }
        prompt = Template(self.prompt_template).render(**inputs)
        log_llm(
            logger=logger,
            log_module="EnterpriseSearchPolicy",
            log_event="enterprise_search_policy._render_prompt.prompt_rendered",
            prompt=prompt,
        )
        return prompt

    async def _generate_llm_answer(
        self, llm: LLMClient, prompt: Text
    ) -> Optional[Text]:
        try:
            llm_response = await llm.acompletion(prompt)
            llm_answer = llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            logger.error(
                "enterprise_search_policy._generate_llm_answer.llm_error",
                error=e,
            )
            llm_answer = None

        return llm_answer

    def _create_prediction(
        self,
        domain: Domain,
        tracker: DialogueStateTracker,
        action_metadata: Dict[Text, Any],
    ) -> PolicyPrediction:
        """Create a policy prediction result with ACTION_SEND_TEXT_NAME.

        Args:
            domain: The model's domain.
            tracker: The tracker containing the conversation history up to now.
            action_metadata: The metadata for the predicted action.

        Returns:
            The prediction.
        """
        result = self._prediction_result(ACTION_SEND_TEXT_NAME, domain)
        stack = tracker.stack
        if not stack.is_empty():
            stack.pop()
            events: List[Event] = tracker.create_stack_updated_events(stack)
        else:
            events = []

        return self._prediction(result, action_metadata=action_metadata, events=events)

    def _create_prediction_internal_error(
        self, domain: Domain, tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        return self._create_prediction_for_pattern(
            domain, tracker, InternalErrorPatternFlowStackFrame()
        )

    def _create_prediction_cannot_handle(
        self, domain: Domain, tracker: DialogueStateTracker
    ) -> PolicyPrediction:
        return self._create_prediction_for_pattern(
            domain, tracker, CannotHandlePatternFlowStackFrame()
        )

    def _create_prediction_for_pattern(
        self,
        domain: Domain,
        tracker: DialogueStateTracker,
        pattern_stack_frame: PatternFlowStackFrame,
    ) -> PolicyPrediction:
        """Create a policy prediction result for error.

        We should cancel the current flow (hence ACTION_CANCEL_FLOW) and push a
        pattern stack frame (Internal Error Pattern by default) to start the pattern.

        Args:
            domain: The model's domain.
            tracker: The tracker containing the conversation history up to now.
            pattern_stack_frame: The pattern stack frame to push.

        Returns:
            The prediction.
        """
        # TODO: replace ACTION_CANCEL_FLOW (ATO-2097)
        result = self._prediction_result(ACTION_CANCEL_FLOW, domain)
        stack = tracker.stack
        if not stack.is_empty():
            stack.pop()
            stack.push(pattern_stack_frame)
        events: List[Event] = tracker.create_stack_updated_events(stack)
        return self._prediction(result, action_metadata=None, events=events)

    def _prediction_result(
        self, action_name: Optional[Text], domain: Domain, score: Optional[float] = 1.0
    ) -> List[float]:
        """Creates a prediction result.

        Args:
            action_name: The name of the predicted action.
            domain: The model's domain.
            score: The score of the predicted action.

        Returns:
        The prediction result where the score is used for one hot encoding.
        """
        result = self._default_predictions(domain)
        if action_name:
            result[domain.index_for_action(action_name)] = score  # type: ignore[assignment]
        return result

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> "EnterpriseSearchPolicy":
        """Loads a trained policy (see parent class for full docstring)."""
        prompt_template = None
        store_type = config.get(VECTOR_STORE_PROPERTY, {}).get(
            VECTOR_STORE_TYPE_PROPERTY
        )

        embeddings = cls._create_plain_embedder(config)
        logger.info("enterprise_search_policy.load", config=config)
        if store_type == DEFAULT_VECTOR_STORE_TYPE:
            # if a vector store is not specified,
            # default to using FAISS with the index stored in the model
            # TODO figure out a way to get path without context manager
            with model_storage.read_from(resource) as path:
                vector_store = FAISS_Store(
                    embeddings=embeddings,
                    index_path=path,
                    docs_folder=None,
                    create_index=False,
                )
        else:
            vector_store = create_from_endpoint_config(
                config_type=store_type,
                embeddings=embeddings,
            )  # type: ignore
        try:
            with model_storage.read_from(resource) as path:
                prompt_template = rasa.shared.utils.io.read_file(
                    path / ENTERPRISE_SEARCH_PROMPT_FILE_NAME
                )

        except (FileNotFoundError, FileNotFoundError) as e:
            logger.warning(
                "enterprise_search_policy.load.failed", error=e, resource=resource.name
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            vector_store=vector_store,
            prompt_template=prompt_template,
        )

    @classmethod
    def _get_local_knowledge_data(cls, config: Dict[str, Any]) -> Optional[List[str]]:
        """This is required only for local knowledge base types.

        e.g. FAISS, to ensure that the graph component is retrained when the knowledge
        base is updated.
        """
        merged_config = {**cls.get_default_config(), **config}

        store_type = merged_config.get(VECTOR_STORE_PROPERTY, {}).get(
            VECTOR_STORE_TYPE_PROPERTY
        )
        if store_type != DEFAULT_VECTOR_STORE_TYPE:
            return None

        source = merged_config.get(VECTOR_STORE_PROPERTY, {}).get(SOURCE_PROPERTY)
        if not source:
            return None

        docs = FAISS_Store.load_documents(source)

        if len(docs) == 0:
            return None

        docs_as_strings = [
            json.dumps(doc.dict(), ensure_ascii=False, sort_keys=True) for doc in docs
        ]
        return sorted(docs_as_strings)

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base and prompt template for the graph."""
        local_knowledge_data = cls._get_local_knowledge_data(config)

        prompt_template = get_prompt_template(
            config.get(PROMPT_CONFIG_KEY),
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint([prompt_template, local_knowledge_data])

    @staticmethod
    def post_process_citations(llm_answer: str) -> str:
        """Post-process the LLM answer.

         Re-writes the bracketed numbers to start from 1 and
         re-arranges the sources to follow the enumeration order.

        Args:
            llm_answer: The LLM answer.

        Returns:
            The post-processed LLM answer.
        """
        logger.debug(
            "enterprise_search_policy.post_process_citations", llm_answer=llm_answer
        )

        # Split llm_answer into answer and citations
        try:
            answer, citations = llm_answer.rsplit("Sources:", 1)
        except ValueError:
            # if there is no "Sources:" in the llm_answer
            return llm_answer

        # Find all source references in the answer
        pattern = r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]"
        matches = re.findall(pattern, answer)
        old_source_indices = [
            int(num.strip()) for match in matches for num in match.split(",")
        ]

        # Map old source references to the correct enumeration
        renumber_mapping = {num: idx + 1 for idx, num in enumerate(old_source_indices)}

        # remove whitespace from original source citations in answer
        for match in matches:
            answer = answer.replace(f"[{match}]", f"[{match.replace(' ', '')}]")

        new_answer = []
        for word in answer.split():
            matches = re.findall(pattern, word)
            if matches:
                for match in matches:
                    if "," in match:
                        old_indices = [
                            int(num.strip()) for num in match.split(",") if num
                        ]
                        new_indices = [
                            renumber_mapping[old_index]
                            for old_index in old_indices
                            if old_index in renumber_mapping
                        ]
                        if not new_indices:
                            continue

                        word = word.replace(
                            match, f"{', '.join(map(str, new_indices))}"
                        )
                    else:
                        old_index = int(match.strip("[].,:;?!"))
                        new_index = renumber_mapping.get(old_index)
                        if not new_index:
                            continue

                        word = word.replace(str(old_index), str(new_index))
            new_answer.append(word)

        # join the words
        joined_answer = " ".join(new_answer)
        joined_answer += "\nSources:\n"

        new_sources: List[str] = []

        for line in citations.split("\n"):
            pattern = r"(?<=\[)\d+"
            match = re.search(pattern, line)
            if match:
                old_index = int(match.group(0))
                new_index = renumber_mapping[old_index]
                # replace only the first occurrence of the old index
                line = line.replace(f"[{old_index}]", f"[{new_index}]", 1)

                # insert the line into the new_index position
                new_sources.insert(new_index - 1, line)
            elif line.strip():
                new_sources.append(line)

        joined_sources = "\n".join(new_sources)

        return joined_answer + joined_sources
