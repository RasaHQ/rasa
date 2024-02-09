import importlib.resources
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text

import dotenv
import rasa.shared.utils.io
import structlog
from jinja2 import Template
from pydantic.error_wrappers import ValidationError
from rasa.core.constants import (
    POLICY_MAX_HISTORY,
    POLICY_PRIORITY,
    SEARCH_POLICY_PRIORITY,
)
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.utils import AvailableEndpoints
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
from rasa.shared.core.constants import (
    ACTION_SEND_TEXT_NAME,
    DEFAULT_SLOT_NAMES,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.cli import print_error_and_exit
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    embedder_factory,
    get_prompt_template,
    llm_factory,
    tracker_as_readable_transcript,
)

from rasa.core.information_retrieval.faiss import FAISS_Store
from rasa.core.information_retrieval.information_retrieval import (
    InformationRetrieval,
    create_from_endpoint_config,
)

if TYPE_CHECKING:
    from langchain.schema.embeddings import Embeddings
    from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer

from rasa.utils.log_utils import log_llm

structlogger = structlog.get_logger()

dotenv.load_dotenv("./.env")

SOURCE_PROPERTY = "source"
VECTOR_STORE_TYPE_PROPERTY = "type"
VECTOR_STORE_PROPERTY = "vector_store"

DEFAULT_VECTOR_STORE_TYPE = "faiss"
DEFAULT_VECTOR_STORE = {
    VECTOR_STORE_TYPE_PROPERTY: DEFAULT_VECTOR_STORE_TYPE,
    SOURCE_PROPERTY: "./docs",
}

DEFAULT_LLM_CONFIG = {
    "_type": "openai",
    "request_timeout": 5,
    "temperature": 0.0,
    "model_name": DEFAULT_OPENAI_CHAT_MODEL_NAME,
}

DEFAULT_EMBEDDINGS_CONFIG = {
    "_type": "openai",
    "model": DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

EMBEDDINGS_CONFIG_KEY = "embeddings"
LLM_CONFIG_KEY = "llm"
ENTERPRISE_SEARCH_PROMPT_FILE_NAME = "enterprise_search_policy_prompt.jinja2"

DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.core.policies", "enterprise_search_prompt_template.jinja2"
)


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
            - name: rasa_plus.ml.EnterpriseSearchPolicy
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

        self.vector_store = vector_store
        self.vector_store_config = config.get(
            VECTOR_STORE_PROPERTY, DEFAULT_VECTOR_STORE
        )
        self.max_history = self.config.get(POLICY_MAX_HISTORY)
        self.prompt_template = prompt_template or get_prompt_template(
            self.config.get("prompt"),
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
        )

    @classmethod
    def _create_plain_embedder(cls, config: Dict[Text, Any]) -> "Embeddings":
        """Creates an embedder based on the given configuration.

        Returns:
        The embedder.
        """
        return embedder_factory(
            config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )

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

        # validate embedding configuration
        try:
            embeddings = self._create_plain_embedder(self.config)
        except ValidationError as e:
            print_error_and_exit(
                "Unable to create embedder. Please make sure you specified the "
                f"required environment variables. Error: {e}"
            )

        # validate llm configuration
        try:
            llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)
        except (ImportError, ValueError, ValidationError) as e:
            # ImportError: llm library is likely not installed
            # ValueError: llm config is likely invalid
            # ValidationError: environment variables are likely not set
            print_error_and_exit(f"Unable to create LLM. Error: {e}")

        if store_type == DEFAULT_VECTOR_STORE_TYPE:
            structlogger.info("enterprise_search_policy.train.faiss")
            with self._model_storage.write_to(self._resource) as path:
                self.vector_store = FAISS_Store(
                    docs_folder=self.vector_store_config.get(SOURCE_PROPERTY),
                    embeddings=embeddings,
                    index_path=path,
                    create_index=True,
                )
        else:
            structlogger.info(
                "enterprise_search_policy.train.custom", store_type=store_type
            )

        self.persist()
        return self._resource

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / ENTERPRISE_SEARCH_PROMPT_FILE_NAME
            )

    def _create_return_message(self, text: Text, domain: Domain) -> PolicyPrediction:
        """Creates a message which can be returned by the policy.

        Args:
            text: The text of the message.
            domain: The model's domain.

        Returns:
            The message.
        """
        message = {
            "text": text,
        }
        result = self._prediction_result(ACTION_SEND_TEXT_NAME, domain)
        return self._prediction(result, action_metadata={"message": message})

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
            structlogger.error(
                "enterprise_search_policy._connect_vector_store_or_raise.no_config"
            )
            raise ValueError(
                """No vector store specified. Please specify a vector
                store in the endpoints configuration"""
            )
        try:
            self.vector_store.connect(config)  # type: ignore
        except Exception as e:
            structlogger.error(
                "enterprise_search_policy._connect_vector_store_or_raise.connect_error",
                error=e,
            )
            raise ValueError(f"Unable to connect to the vector store. Error: {e}")

    def predict_action_probabilities(  # type: ignore[override]
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
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)
        if not self.supports_current_stack_frame(tracker, False, False):
            return self._prediction(self._default_predictions(domain))

        if self.vector_store is not None:
            search_query = tracker_as_readable_transcript(tracker, max_turns=1)
            self._connect_vector_store_or_raise(endpoints)
            documents = self.vector_store.search(search_query)
            inputs = {
                "current_conversation": tracker_as_readable_transcript(
                    tracker, max_turns=self.max_history
                ),
                "docs": documents,
                "slots": self._prepare_slots_for_template(tracker),
            }
            prompt = Template(self.prompt_template).render(**inputs)
            log_llm(
                logger=structlogger,
                log_module="EnterpriseSearchPolicy",
                log_event="enterprise_search_policy.predict_action_probabilities.prompt_rendered",
                prompt=prompt,
            )
            try:
                llm_answer = llm(prompt)
            except Exception as e:
                # unfortunately, langchain does not wrap LLM exceptions which means
                # we have to catch all exceptions here
                structlogger.error("nlg.llm.error", error=e)
                llm_answer = None
        else:
            structlogger.error(
                "enterprise_search_policy.predict_action_probabilities.no_vector_store"
            )
            llm_answer = None

        structlogger.debug(
            "enterprise_search_policy.predict_action_probabilities.llm_answer",
            llm_answer=llm_answer,
        )
        if llm_answer:
            predicted_action_name = ACTION_SEND_TEXT_NAME
            action_metadata = {
                "message": {
                    "text": llm_answer,
                }
            }
        else:
            predicted_action_name = None
            action_metadata = None

        structlogger.debug(
            "enterprise_search_policy.predict_action_probabilities.predicted_action_name",
            predicted_action_name=predicted_action_name,
        )
        result = self._prediction_result(predicted_action_name, domain)

        stack = tracker.stack
        if not stack.is_empty():
            stack.pop()
            events: List[Event] = tracker.create_stack_updated_events(stack)
        else:
            events = []

        return self._prediction(result, action_metadata=action_metadata, events=events)

    def _prediction_result(
        self, action_name: Optional[Text], domain: Domain, score: Optional[float] = 1.0
    ) -> List[float]:
        """Creates a prediction result.

        Args:
            action_name: The name of the predicted action.
            domain: The model's domain.
            score: The score of the predicted action.

        Resturns:
        The prediction result where the score is used for one hot encoding.
        """
        result = self._default_predictions(domain)
        if action_name:
            result[domain.index_for_action(action_name)] = score  # type: ignore[assignment]  # noqa: E501
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
        structlogger.info("enterprise_search_policy.load", config=config)
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
            structlogger.warning(
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
            config.get("prompt"),
            DEFAULT_ENTERPRISE_SEARCH_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint([prompt_template, local_knowledge_data])
