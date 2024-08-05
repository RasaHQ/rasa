from __future__ import annotations

from typing import Any, Dict, List, Optional, Text

import rasa.shared.utils.io
import structlog
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS

from rasa import telemetry
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.shared.exceptions import FileIOException
from rasa.shared.nlu.constants import (
    INTENT,
    INTENT_NAME_KEY,
    INTENT_RESPONSE_KEY,
    METADATA,
    PREDICTED_CONFIDENCE_KEY,
    TEXT,
)
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    DEFAULT_OPENAI_TEMPERATURE,
    combine_custom_and_default_config,
    embedder_factory,
    get_prompt_template,
    llm_factory,
)

from rasa.utils.ml_utils import (
    load_faiss_vector_store,
    persist_faiss_vector_store,
)
from rasa.utils import beta

structlogger = structlog.get_logger()

RASA_PRO_BETA_LLM_INTENT = "RASA_PRO_BETA_LLM_INTENT"

DEFAULT_NUMBER_OF_INTENT_EXAMPLES = 10

DEFAULT_LLM_CONFIG = {
    "api_type": "openai",
    "model": DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    "request_timeout": 5,
    "temperature": DEFAULT_OPENAI_TEMPERATURE,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
}

DEFAULT_EMBEDDINGS_CONFIG = {"_type": "openai"}

EMBEDDINGS_CONFIG_KEY = "embeddings"
LLM_CONFIG_KEY = "llm"

DEFAULT_INTENT_CLASSIFICATION_PROMPT_TEMPLATE = """Label a users message from a
conversation with an intent. Reply ONLY with the name of the intent.

The intent should be one of the following:
{% for intent in intents %}- {{intent}}
{% endfor %}
{% for example in examples %}
Message: {{example['text']}}
Intent: {{example['intent']}}
{% endfor %}
Message: {{message}}
Intent:"""

LLM_INTENT_CLASSIFIER_PROMPT_FILE_NAME = "llm_intent_classifier_prompt.jinja2"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True
)
class LLMIntentClassifier(GraphComponent, IntentClassifier):
    """Intent classifier using the LLM to generate the intent classification."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            "fallback_intent": "out_of_scope",
            "prompt": None,
            LLM_CONFIG_KEY: None,
            EMBEDDINGS_CONFIG_KEY: None,
            "number_of_examples": DEFAULT_NUMBER_OF_INTENT_EXAMPLES,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        intent_docsearch: Optional[FAISS] = None,
        example_docsearch: Optional[FAISS] = None,
        available_intents: Optional[List[str]] = None,
        prompt_template: Optional[Text] = None,
    ) -> None:
        """Creates classifier."""
        beta.ensure_beta_feature_is_enabled(
            "LLM Intent Classifier", env_flag=RASA_PRO_BETA_LLM_INTENT
        )

        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource
        self._execution_context = execution_context

        self.fallback_intent = self.component_config.get("fallback_intent")
        self.number_of_examples = self.component_config.get(
            "number_of_examples", DEFAULT_NUMBER_OF_INTENT_EXAMPLES
        )

        self.intent_docsearch = intent_docsearch
        self.example_docsearch = example_docsearch
        self.available_intents = set(available_intents or [])
        self.prompt_template = prompt_template or get_prompt_template(
            self.component_config.get("prompt"),
            DEFAULT_INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
        )

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> LLMIntentClassifier:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    def embeddings_property(self, prop: str) -> Optional[str]:
        """Returns the property of the embeddings config."""
        return combine_custom_and_default_config(
            self.component_config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        ).get(prop)

    def llm_property(self, prop: str) -> Optional[str]:
        """Returns the property of the LLM config."""
        return combine_custom_and_default_config(
            self.component_config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
        ).get(prop)

    def custom_prompt_template(self) -> Optional[str]:
        """Returns the custom prompt template if it is not the default one."""
        if self.prompt_template != DEFAULT_INTENT_CLASSIFICATION_PROMPT_TEMPLATE:
            return self.prompt_template
        else:
            return None

    def train(self, training_data: TrainingData) -> Resource:
        """Trains the intent classifier on a data set."""
        texts = [ex.get(TEXT, "") for ex in training_data.intent_examples]
        metadatas = [
            {
                INTENT: ex.get(INTENT_RESPONSE_KEY) or ex.get(INTENT),
                TEXT: ex.get(TEXT),
            }
            for ex in training_data.intent_examples
        ]

        embedder = embedder_factory(
            self.component_config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )

        self.example_docsearch = (
            FAISS.from_texts(texts, embedder, metadatas) if texts else None
        )

        self.available_intents = {e[INTENT].lower() for e in metadatas}

        self.intent_docsearch = (
            FAISS.from_texts(
                list(self.available_intents),
                embedder,
            )
            if self.available_intents
            else None
        )

        self.persist()
        telemetry.track_llm_intent_train_completed(
            embeddings_type=self.embeddings_property("_type"),
            embeddings_model=self.embeddings_property("model_name")
            or self.embeddings_property("model"),
            llm_type=self.llm_property("_type"),
            llm_model=self.llm_property("model_name") or self.llm_property("model"),
            fallback_intent=self.fallback_intent,
            custom_prompt_template=self.custom_prompt_template(),
            number_of_examples=self.number_of_examples,
            number_of_available_intents=len(self.available_intents),
        )
        return self._resource

    async def process(self, messages: List[Message]) -> List[Message]:
        """Sets the message intent and add it to the output if it exists."""
        for message in messages:
            if message.get(TEXT, "").startswith(INTENT_MESSAGE_PREFIX):
                # llm calls are expensive, so we skip messages
                # that start with a slash as they are direct intents
                continue
            examples = self.select_few_shot_examples(message)
            predicted_intent_name = await self.classify_intent_of_message(
                message, examples
            )

            if not predicted_intent_name:
                # or should we predict self.fallback_intent?
                continue

            if "/" in predicted_intent_name:
                intent_name = predicted_intent_name.split("/")[0]
            else:
                intent_name = predicted_intent_name

            structlogger.info(
                "llmintent.prediction.success",
                predicted_intent=intent_name,
                llm_prediction=predicted_intent_name,
            )

            intent = {
                INTENT_NAME_KEY: intent_name,
                METADATA: {"llm_intent": predicted_intent_name},
                PREDICTED_CONFIDENCE_KEY: 1.0,
            }

            message.set(INTENT, intent, add_to_output=True)

        # telemetry.track_llm_intent_predict(
        #     embeddings_type=self.embeddings_property("_type"),
        #     embeddings_model=self.embeddings_property("model_name")
        #     or self.embeddings_property("model"),
        #     llm_type=self.llm_property("_type"),
        #     llm_model=self.llm_property("model_name") or self.llm_property("model"),
        # )
        return messages

    async def _generate_llm_response(
        self, message: Message, intents: List[str], few_shot_examples: List[Document]
    ) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            message: The message.
            intents: The intents.
            few_shot_examples: The few shot examples.

        Returns:
            generated text
        """
        prompt = self.get_prompt_template_obj()
        examples = [
            {**{"text": e.page_content}, **e.metadata} for e in few_shot_examples
        ]
        message_text = message.get(TEXT, "")

        structlogger.debug(
            "llmintent.llm.generate",
            prompt=prompt.format(
                examples=examples,
                intents=intents,
                message=message_text,
            ),
        )
        chain = LLMChain(
            llm=llm_factory(
                self.component_config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
            ),
            prompt=self.get_prompt_template_obj(),
        )

        try:
            return await chain.arun(
                examples=examples, intents=intents, message=message_text
            )
        except Exception as e:
            structlogger.error("llmintent.llm.error", error=e)
            return None

    def closest_intent_from_training_data(self, generated_intent: str) -> Optional[str]:
        """Returns the closest intent from the training data.

        Args:
            generated_intent: the intent that was generated by the LLM

        Returns:
            the closest intent from the training data.
        """
        structlogger.debug(
            "llmintent.llmresponse.not_in_training_data",
            proposed_intent=generated_intent,
        )
        if not self.intent_docsearch:
            return None

        try:
            closest_intents = (
                self.intent_docsearch.similarity_search_with_relevance_scores(
                    generated_intent, k=1
                )
            )
        except Exception as e:
            structlogger.error("llmintent.llmresponse.intentsearch.error", error=e)
            return None

        if not closest_intents:
            return None

        closest_intent_name = closest_intents[0][0].page_content

        structlogger.debug(
            "llmintent.llmresponse.intentsearch",
            intent=closest_intent_name,
            score=closest_intents[0][1],
        )
        return closest_intent_name

    def get_prompt_template_obj(self) -> PromptTemplate:
        """Returns the prompt template."""
        return PromptTemplate(
            input_variables=["examples", "intents", "message"],
            template=self.prompt_template,
            template_format="jinja2",
        )

    async def classify_intent_of_message(
        self, message: Message, few_shot_examples: List[Document]
    ) -> Optional[Text]:
        """Classify the message using an LLM.

        Args:
            message: The message to classify.
            few_shot_examples: The few shot examples that can be used in the prompt.


        Returns:
        The predicted intent.
        """
        provided_intents = self.select_intent_examples(message, few_shot_examples)

        generated_intent = await self._generate_llm_response(
            message, provided_intents, few_shot_examples
        )

        if generated_intent is None:
            # something went wrong with the LLM
            return None

        generated_intent = generated_intent.strip().lower()

        if generated_intent in self.available_intents:
            return generated_intent

        # if the generated intent is not in the training data, we try to
        # find the closest intent
        return self.closest_intent_from_training_data(generated_intent)

    def persist(self) -> None:
        """Persist this model into the passed directory."""
        with self._model_storage.write_to(self._resource) as path:
            persist_faiss_vector_store(path / "intents_faiss", self.intent_docsearch)
            persist_faiss_vector_store(path / "examples_faiss", self.example_docsearch)
            rasa.shared.utils.io.dump_obj_as_json_to_file(
                path / "intents.json", list(self.available_intents)
            )
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / LLM_INTENT_CLASSIFIER_PROMPT_FILE_NAME
            )

    def select_intent_examples(
        self, message: Message, few_shot_examples: List[Document]
    ) -> List[str]:
        """Returns the intents that are used in the classification prompt.

        The intents are included in the prompt to help the LLM to generate the
        correct intent. The selected intents can be based on the message or on
        the few shot examples which are also included in the prompt.

        Including all intents can lead to a very long prompt which will lead
        to higher costs and longer response times. In addition, the LLM might
        not be able to generate the correct intent if there are too many intents
        in the prompt as we can't include an example for every intent. The
        classification would in this case just be based on the intent name.

        Args:
            message: The message to classify.
            few_shot_examples: The few shot examples that can be used in the prompt.


        Returns:
        The intents that are used in the classification prompt.
        """
        # we sort the list to make sure intents are always in the same order
        # independent of the order of the examples
        selected_intents = sorted(
            list(dict.fromkeys([e.metadata["intent"] for e in few_shot_examples]))
        )
        if selected_intents:
            return selected_intents
        else:
            return list(self.available_intents)[:5]

    def select_few_shot_examples(self, message: Message) -> List[Document]:
        """Selects the few shot examples that should be used for the LLM prompt.

        The examples are included in the classification prompt to help the LLM
        to generate the correct intent. Since only a few examples are included
        in the prompt, we need to select the most relevant ones.

        Args:
            message: the message to find the closest examples for

        Returns:
            the closest examples from the embedded training data
        """
        if not self.example_docsearch:
            return []

        # we fetch more examples than we need to make sure that we have enough
        # examples that are relevant to the message. the forumla ensures
        # that we fetch at least double the number of examples but avoids
        # fetching too many additional examples if the number of
        # examples is large.

        fetch_k = int(self.number_of_examples * (2 + 10 * 1 / self.number_of_examples))

        try:
            return self.example_docsearch.max_marginal_relevance_search(
                message.get(TEXT, ""), k=self.number_of_examples, fetch_k=fetch_k
            )
        except Exception as e:
            # this can happen if the message doesn't have a text attribute
            structlogger.warning(
                "llmintent.embeddong.no_embedding", message=message, error=e
            )
            return []

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> LLMIntentClassifier:
        """Loads trained component (see parent class for full docstring).

        Args:
            config: the configuration for this component
            model_storage: the model storage
            resource: the resource
            execution_context: the execution context
            **kwargs: additional arguments


        Returns:
        the loaded component
        """
        example_docsearch = None
        intent_docsearch = None
        available_intents = None
        prompt_template = None

        embedder = embedder_factory(
            config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )
        try:
            with model_storage.read_from(resource) as path:
                example_docsearch = load_faiss_vector_store(
                    path / "examples_faiss", embedder
                )
                intent_docsearch = load_faiss_vector_store(
                    path / "intents_faiss", embedder
                )
                available_intents = rasa.shared.utils.io.read_json_file(
                    path / "intents.json"
                )
                prompt_template = rasa.shared.utils.io.read_file(
                    path / LLM_INTENT_CLASSIFIER_PROMPT_FILE_NAME
                )

        except (ValueError, FileNotFoundError, FileIOException) as e:
            structlogger.warning(
                "llmintent.load.failed", error=e, resource=resource.name
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            intent_docsearch,
            example_docsearch,
            available_intents,
            prompt_template,
        )

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base for the graph."""
        prompt_template = get_prompt_template(
            config.get("prompt"),
            DEFAULT_INTENT_CLASSIFICATION_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint(prompt_template)
