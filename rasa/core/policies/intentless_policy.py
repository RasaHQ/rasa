import importlib.resources
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING, Text, Tuple

import structlog
import tiktoken
from jinja2 import Template
from langchain.docstore.document import Document
from langchain.schema.embeddings import Embeddings
from langchain_community.vectorstores.faiss import FAISS

import rasa.shared.utils.io
from rasa import telemetry
from rasa.core.constants import (
    CHAT_POLICY_PRIORITY,
    POLICY_PRIORITY,
    UTTER_SOURCE_METADATA_KEY,
)
from rasa.core.policies.policy import Policy, PolicyPrediction, SupportedData
from rasa.dialogue_understanding.stack.frames import (
    ChitChatStackFrame,
    DialogueStackFrame,
)
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.graph_components.providers.forms_provider import Forms
from rasa.graph_components.providers.responses_provider import Responses
from rasa.shared.constants import (
    REQUIRED_SLOTS_KEY,
    EMBEDDINGS_CONFIG_KEY,
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    PROMPT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    OPENAI_PROVIDER,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.constants import ACTION_LISTEN_NAME
from rasa.shared.core.domain import KEY_RESPONSES_TEXT, Domain
from rasa.shared.core.events import (
    ActionExecuted,
    BotUttered,
    Event,
    UserUttered,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.exceptions import FileIOException, RasaCoreException
from rasa.shared.nlu.constants import PREDICTED_CONFIDENCE_KEY
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.providers.embedding._langchain_embedding_client_adapter import (
    _LangchainEmbeddingClientAdapter,
)
from rasa.shared.providers.llm.llm_client import LLMClient
from rasa.shared.utils.io import deep_container_fingerprint
from rasa.shared.utils.llm import (
    AI,
    DEFAULT_OPENAI_CHAT_MODEL_NAME,
    DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    USER,
    combine_custom_and_default_config,
    embedder_factory,
    get_prompt_template,
    llm_factory,
    sanitize_message_for_prompt,
    tracker_as_readable_transcript,
    try_instantiate_llm_client,
)
from rasa.utils.ml_utils import (
    extract_ai_response_examples,
    extract_participant_messages_from_transcript,
    form_utterances_to_action,
    load_faiss_vector_store,
    persist_faiss_vector_store,
    response_for_template,
)
from rasa.dialogue_understanding.patterns.chitchat import FLOW_PATTERN_CHITCHAT
from rasa.shared.core.constants import ACTION_TRIGGER_CHITCHAT
from rasa.utils.log_utils import log_llm

if TYPE_CHECKING:
    from rasa.core.featurizers.tracker_featurizers import TrackerFeaturizer

structlogger = structlog.get_logger()

NUMBER_OF_CONVERSATION_SAMPLES = 5

NUMBER_OF_RESPONSE_SAMPLES = 20

# GPT-3 model's MAX_TOKEN = 2049
# https://platform.openai.com/docs/models/gpt-3
# get_response_prompt empty template uses around 149 tokens
# equal number of tokens for conversation and responses
# (2049 - 149) / 2 = 950
# 100 tokens for input
MAX_NUMBER_OF_TOKENS_FOR_SAMPLES = 900

# the config property name for the confidence of the nlu prediction
NLU_ABSTENTION_THRESHOLD = "nlu_abstention_threshold"

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_CHAT_MODEL_NAME,
    "temperature": 0.0,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 5,
}

DEFAULT_EMBEDDINGS_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    "model": DEFAULT_OPENAI_EMBEDDING_MODEL_NAME,
}

DEFAULT_INTENTLESS_PROMPT_TEMPLATE = importlib.resources.open_text(
    "rasa.core.policies", "intentless_prompt_template.jinja2"
).name

INTENTLESS_PROMPT_TEMPLATE_FILE_NAME = "intentless_policy_prompt.jinja2"


class RasaMLPolicyTrainingException(RasaCoreException):
    """Raised when training fails."""

    pass


@dataclass
class Interaction:
    text: str
    actor: str


@dataclass
class Conversation:
    interactions: List[Interaction] = field(default_factory=list)


def collect_form_responses(forms: Forms) -> Set[Text]:
    """Collect responses that belong the requested slots in forms.

    Args:
        forms: the forms from the domain
    Returns:
        all utterances used in forms
    """
    form_responses = set()
    for _, form_info in forms.data.items():
        for required_slot in form_info.get(REQUIRED_SLOTS_KEY, []):
            form_responses.add(f"utter_ask_{required_slot}")
    return form_responses


def filter_responses(responses: Responses, forms: Forms, flows: FlowsList) -> Responses:
    """Filters out responses that are unwanted for the intentless policy.

    This includes utterances used in flows and forms.

    Args:
        responses: the responses from the domain
        forms: the forms from the domain
        flows: all flows
    Returns:
        The remaining, relevant responses for the intentless policy.
    """
    form_responses = collect_form_responses(forms)
    flow_responses = flows.utterances
    combined_responses = form_responses | flow_responses
    filtered_responses = {
        name: variants
        for name, variants in responses.data.items()
        if name not in combined_responses
    }

    pattern_chitchat = flows.flow_by_id(FLOW_PATTERN_CHITCHAT)

    # The following condition is highly unlikely, but mypy requires the case
    # of pattern_chitchat == None to be addressed
    if not pattern_chitchat:
        return Responses(data=filtered_responses)

    # if action_trigger_chitchat, filter out "utter_free_chitchat_response"
    has_action_trigger_chitchat = pattern_chitchat.has_action_step(
        ACTION_TRIGGER_CHITCHAT
    )
    if has_action_trigger_chitchat:
        filtered_responses.pop("utter_free_chitchat_response", None)

    return Responses(data=filtered_responses)


def action_from_response(
    text: Optional[str], responses: Dict[Text, List[Dict[Text, Any]]]
) -> Optional[str]:
    """Returns the action associated with the given response text.

    Args:
        text: The response text.
        responses: The responses from the domain.

    Returns:
    The action associated with the response text, or None if no action is found.
    """
    if not text:
        return None

    for action, variations in responses.items():
        for variation in variations:
            if variation.get(KEY_RESPONSES_TEXT) == text:
                return action
    return None


def _conversation_sample_from_tracker(
    tracker: DialogueStateTracker,
    responses: Dict[Text, List[Dict[Text, Any]]],
) -> Optional[Conversation]:
    """Extracts a conversation sample from the given tracker.

    Example:
        >>> tracker = DialogueStateTracker(
        ...     "default",
        ...     slots=[],
        ...     active_loop={},
        ...     events=[
        ...         UserUttered("hello"),
        ...         BotUttered("hey there!"),
        ...         UserUttered("goodbye"),
        ...         BotUttered("bye bye!"),
        ...     ],
        ... )
        >>> _conversation_sample_from_tracker(tracker)
        Conversation(interactions=[
            Interaction(actor='user', text='hello'),
            Interaction(actor='bot', text='hey there!'),
            Interaction(actor='user', text='goodbye'),
            Interaction(actor='bot', text='bye bye!'),
        ])

    Args:
        tracker: The tracker to extract a conversation sample from.
        responses: The responses from the domain.

    Returns:
        The conversation sample, or None if the tracker doesn't contain a
    conversation sample.
    """
    conversation = Conversation()
    for event in tracker.applied_events():
        if isinstance(event, UserUttered):
            if event.text is None:
                # need to abort here, as we can't use this conversation
                # sample as it doesn't have an actual user message attached
                # likely, this is just an intent.
                return None
            conversation.interactions.append(Interaction(actor=USER, text=event.text))
        elif isinstance(event, BotUttered):
            conversation.interactions.append(
                Interaction(actor=AI, text=event.text or "")
            )
        elif (
            isinstance(event, ActionExecuted)
            and event.action_name
            and event.action_name.startswith("utter_")
        ):
            response = response_for_template(event.action_name, responses)
            if response:
                interaction = Interaction(actor=AI, text=response)
                conversation.interactions.append(interaction)
            else:
                # need to abort here, as we can't use this conversation
                # sample as it doesn't have a response for the utterance
                # that was used in the story.
                return None
    # Do not use reduced conversations with just one or no utterances
    if len(conversation.interactions) < 2:
        structlogger.debug(
            "intentless_policy.sampled_conversation.skipped",
            interations=conversation.interactions,
        )
        return None
    # Do not use conversation with only user or only bot
    if len({m.actor for m in conversation.interactions}) < 2:
        structlogger.debug(
            "intentless_policy.sampled_conversation.skipped",
            interactions=conversation.interactions,
        )
        return None
    return conversation


def conversation_samples_from_trackers(
    training_trackers: List[TrackerWithCachedStates],
    responses: Dict[Text, List[Dict[Text, Any]]],
) -> List[Conversation]:
    """Extracts conversation samples from the given trackers.

    Args:
        training_trackers: The trackers to extract conversation samples from.
        responses: The responses from the domain.

    Returns:
    The conversation samples.
    """
    return [
        sample
        for tracker in training_trackers
        if (sample := _conversation_sample_from_tracker(tracker, responses))
    ]


def truncate_documents(
    docs: List[Document],
    max_number_of_tokens: int,
    model_name: str = DEFAULT_OPENAI_CHAT_MODEL_NAME,
) -> List[Document]:
    """Takes first n documents that contains less then `max_number_of_tokens` tokens.

    Args:
        docs: Sequence of documents that needs to be truncated.
        max_number_of_tokens: Maximum number of tokens to preserve.
        model_name: Name of the model to use for tokenization.

    Returns:
        Sequence of documents that contains less then `max_number_of_tokens` tokens.
    """
    enc = tiktoken.encoding_for_model(model_name)

    truncated_docs = []
    docs_token_num = 0
    for doc in docs:
        doc_token_num = len(enc.encode(doc.page_content))
        if docs_token_num + doc_token_num > max_number_of_tokens:
            break

        docs_token_num += doc_token_num
        truncated_docs.append(doc)

    return truncated_docs


def conversation_as_prompt(conversation: Conversation) -> str:
    """Converts the given conversation to a prompt.

    Example:
        >>> conversation = Conversation(interactions=[
        ... Interaction(actor=USER, text="hi"),
        ... Interaction(actor=AI, text="Hello World!")
        ... ])
        >>> print(conversation_as_prompt(conversation))
        USER: hi
        AI: Hello World!

    Args:
        conversation: The conversation to convert.

    Returns:
    The prompt.
    """
    return "\n".join(
        f"{m.actor}: {sanitize_message_for_prompt(m.text)}"
        for m in conversation.interactions
    )


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class IntentlessPolicy(Policy):
    """Policy which uses a language model to generate the next action.

    The policy uses the OpenAI API to generate the next action based on the
    conversation history. The policy can be used to generate a response for
    the user or to predict the next action.
    """

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            POLICY_PRIORITY: CHAT_POLICY_PRIORITY,
            # the abstention threshold is used to determine whether the policy
            # should predict an action or abstain from prediction. if the
            # nlu predictions confidence is above the threshold, the
            # policy will predict with a score lower than the threshold. this
            # ensures that the policy will not override a deterministic policy
            # which utilizes the nlu predictions confidence (e.g. Memoization).
            NLU_ABSTENTION_THRESHOLD: 0.9,
            LLM_CONFIG_KEY: DEFAULT_LLM_CONFIG,
            EMBEDDINGS_CONFIG_KEY: DEFAULT_EMBEDDINGS_CONFIG,
            PROMPT_CONFIG_KEY: DEFAULT_INTENTLESS_PROMPT_TEMPLATE,
        }

    @staticmethod
    def supported_data() -> SupportedData:
        """The type of data supported by this policy.

        By default, this is only ML-based training data. If policies support rule data,
        or both ML-based data and rule data, they need to override this method.

        Returns:
            The data type supported by this policy (ML-based training data).
        """
        return SupportedData.ML_DATA

    @staticmethod
    def does_support_stack_frame(frame: DialogueStackFrame) -> bool:
        """Checks if the policy supports the given stack frame."""
        return isinstance(frame, ChitChatStackFrame)

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        featurizer: Optional["TrackerFeaturizer"] = None,
        responses_docsearch: Optional["FAISS"] = None,
        samples_docsearch: Optional["FAISS"] = None,
        prompt_template: Optional[Text] = None,
    ) -> None:
        """Constructs a new Policy object."""
        super().__init__(config, model_storage, resource, execution_context, featurizer)

        self.nlu_abstention_threshold: float = self.config[NLU_ABSTENTION_THRESHOLD]
        self.response_index = responses_docsearch
        self.conversation_samples_index = samples_docsearch
        self.embedder = self._create_plain_embedder(config)
        self.prompt_template = prompt_template or rasa.shared.utils.io.read_file(
            self.config[PROMPT_CONFIG_KEY]
        )
        self.trace_prompt_tokens = self.config.get("trace_prompt_tokens", False)

    @classmethod
    def _create_plain_embedder(cls, config: Dict[Text, Any]) -> Embeddings:
        """Creates an embedder that uses the OpenAI API.

        Returns:
        The embedder.
        """
        client = embedder_factory(
            config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        )
        return _LangchainEmbeddingClientAdapter(client)

    def embeddings_property(self, prop: str) -> Optional[str]:
        """Returns the property of the embeddings config."""
        return combine_custom_and_default_config(
            self.config.get(EMBEDDINGS_CONFIG_KEY), DEFAULT_EMBEDDINGS_CONFIG
        ).get(prop)

    def llm_property(self, prop: str) -> Optional[str]:
        """Returns the property of the LLM config."""
        return combine_custom_and_default_config(
            self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
        ).get(prop)

    def train(  # type: ignore[override]
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        responses: Responses,
        forms: Forms,
        training_data: TrainingData,
        flows: Optional[FlowsList],
        **kwargs: Any,
    ) -> Resource:
        """Trains a policy.

        Args:
            training_trackers: The story and rules trackers from the training data.
            domain: The model's domain.
            responses: The model's responses.
            forms: The model's forms.
            training_data: The model's training data.
            flows: all existing flows for task-oriented processes
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to train itself.

        Returns:
            A policy must return its resource locator so that potential children nodes
            can load the policy from the resource.
        """
        try_instantiate_llm_client(
            self.config.get(LLM_CONFIG_KEY),
            DEFAULT_LLM_CONFIG,
            "intentless_policy.train",
            "IntentlessPolicy",
        )

        responses = filter_responses(responses, forms, flows or FlowsList([]))
        telemetry.track_intentless_policy_train()
        response_texts = [r for r in extract_ai_response_examples(responses.data)]

        selected_trackers = [
            t for t in training_trackers if not t.is_augmented and not t.is_rule_tracker
        ]

        conversation_samples = conversation_samples_from_trackers(
            selected_trackers, responses.data
        )
        conversations = [
            prompt
            for c in conversation_samples
            if (prompt := conversation_as_prompt(c))
        ]

        try:
            self.response_index = (
                FAISS.from_texts(response_texts, self.embedder, normalize_L2=True)
                if response_texts
                else None
            )

            self.conversation_samples_index = (
                FAISS.from_texts(conversations, self.embedder)
                if conversations
                else None
            )
        except Exception as e:
            structlogger.error(
                "intentless_policy.train.llm.error",
                error=e,
            )
            raise RasaMLPolicyTrainingException(
                "The training of the Intentless Policy failed due to an error "
                "with the LLM provider. Sorry about that! "
                "You can retry your request."
            )

        structlogger.info("intentless_policy.training.completed")
        telemetry.track_intentless_policy_train_completed(
            embeddings_type=self.embeddings_property(PROVIDER_CONFIG_KEY),
            embeddings_model=self.embeddings_property(MODEL_CONFIG_KEY)
            or self.embeddings_property(MODEL_NAME_CONFIG_KEY),
            llm_type=self.llm_property(PROVIDER_CONFIG_KEY),
            llm_model=self.llm_property(MODEL_CONFIG_KEY)
            or self.llm_property(MODEL_NAME_CONFIG_KEY),
        )

        self.persist()
        return self._resource

    def persist(self) -> None:
        """Persists the policy to storage."""
        with self._model_storage.write_to(self._resource) as path:
            persist_faiss_vector_store(path / "responses_faiss", self.response_index)
            persist_faiss_vector_store(
                path / "samples_faiss", self.conversation_samples_index
            )
            rasa.shared.utils.io.write_text_file(
                self.prompt_template, path / INTENTLESS_PROMPT_TEMPLATE_FILE_NAME
            )

    async def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        rule_only_data: Optional[Dict[Text, Any]] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: The tracker containing the conversation history up to now.
            domain: The model's domain.
            rule_only_data: Slots and loops which are specific to rules and hence
                should be ignored by this policy.
            **kwargs: Depending on the specified `needs` section and the resulting
                graph structure the policy can use different input to make predictions.

        Returns:
             The prediction.
        """
        if not self.supports_current_stack_frame(
            tracker
        ) or self.should_abstain_in_coexistence(tracker, True):
            return self._prediction(self._default_predictions(domain))

        if tracker.has_bot_message_after_latest_user_message():
            # if the last event was a bot utterance, we either want to
            # return to the active loop or else predict action_listen as
            # this is the end of the turn
            if tracker.active_loop_name:
                result = self._prediction_result(tracker.active_loop_name, domain)
            else:
                result = self._prediction_result(ACTION_LISTEN_NAME, domain)
            return self._prediction(result)

        if not self.response_index:
            # we don't have any responses, so we can't predict anything
            result = self._default_predictions(domain)
            return self._prediction(result)

        response, score = await self.find_closest_response(tracker)

        predicted_action_name = action_from_response(response, domain.responses)

        form_utterances = form_utterances_to_action(domain)
        if predicted_action_name in form_utterances:
            # if the predicted action is a form utterance, we need to predict the form
            # action instead
            predicted_action_name = form_utterances[predicted_action_name]

        structlogger.info(
            "intentless_policy.prediction.completed",
            predicted_action_name=predicted_action_name,
            score=score,
        )

        telemetry.track_intentless_policy_predict(
            embeddings_type=self.embeddings_property(PROVIDER_CONFIG_KEY),
            embeddings_model=self.embeddings_property(MODEL_CONFIG_KEY)
            or self.embeddings_property(MODEL_NAME_CONFIG_KEY),
            llm_type=self.llm_property(PROVIDER_CONFIG_KEY),
            llm_model=self.llm_property(MODEL_CONFIG_KEY)
            or self.llm_property(MODEL_NAME_CONFIG_KEY),
            score=score,
        )

        result = self._prediction_result(predicted_action_name, domain, score)

        stack = tracker.stack
        if not stack.is_empty():
            stack.pop()
            events: List[Event] = tracker.create_stack_updated_events(stack)
        else:
            events = []

        action_metadata = {UTTER_SOURCE_METADATA_KEY: self.__class__.__name__}

        return self._prediction(result, events=events, action_metadata=action_metadata)

    async def generate_answer(
        self,
        response_examples: List[str],
        conversation_samples: List[str],
        history: str,
    ) -> Optional[str]:
        """Make the llm call to generate an answer."""
        llm = llm_factory(self.config[LLM_CONFIG_KEY], DEFAULT_LLM_CONFIG)
        inputs = {
            "conversations": conversation_samples,
            "responses": response_examples,
            "current_conversation": history,
        }
        prompt = Template(self.prompt_template).render(**inputs)
        log_llm(
            logger=structlogger,
            log_module="IntentlessPolicy",
            log_event="intentless_policy.generate_answer.prompt_rendered",
            prompt=prompt,
        )
        return await self._generate_llm_answer(llm, prompt)

    async def _generate_llm_answer(self, llm: LLMClient, prompt: str) -> Optional[str]:
        try:
            llm_response = await llm.acompletion(prompt)
            return llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("intentless_policy.answer_generation.failed", error=e)
            return None

    def embed_llm_response(self, llm_response: str) -> Optional[List[float]]:
        """Embed the llm response."""
        try:
            # using embed documents here because the responses are the documents
            return self.embedder.embed_documents([llm_response])[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("intentless_policy.answer_embedding.failed", error=e)
            return None

    async def find_closest_response(
        self, tracker: DialogueStateTracker
    ) -> Tuple[Optional[str], float]:
        """Find the closest response fitting the conversation in the tracker.

        Generates a response using the OpenAI API and then finds the closest
        response in the domains responses. The closest response is determined
        using embeddings of the response and all the responses in the domain.
        The embedding is generated using the OpenAI API and the HyDE model.

        Args:
            tracker: The tracker containing the conversation history up to now.
            policy_model: The model's persisted data.
            company: The company the model is used for.

        Returns:
            The response and the score of the response.
        """
        if not self.response_index:
            structlogger.debug("intentless_policy.prediction.skip_noresponses")
            return None, 0.0

        if not tracker.latest_message or not tracker.latest_message.text:
            # we can't generate a response if there is no text on the latest
            # user message
            structlogger.debug("intentless_policy.prediction.skip_notext")
            return None, 0.0

        if tracker.latest_message.text.startswith("/"):
            # we don't want to generate a response if the user is trying to
            # execute a "command" - this should be handled by the regex
            # intent classifier in rasa pro.
            structlogger.debug("intentless_policy.prediction.skip_slash")
            return None, 0.0

        history = tracker_as_readable_transcript(tracker)
        ai_response_examples = self.select_response_examples(
            history,
            number_of_samples=NUMBER_OF_RESPONSE_SAMPLES,
            max_number_of_tokens=MAX_NUMBER_OF_TOKENS_FOR_SAMPLES,
        )
        conversation_samples = self.select_few_shot_conversations(
            history,
            number_of_samples=NUMBER_OF_CONVERSATION_SAMPLES,
            max_number_of_tokens=MAX_NUMBER_OF_TOKENS_FOR_SAMPLES,
        )

        extra_ai_responses = self.extract_ai_responses(conversation_samples)

        # put conversation responses in front of sampled examples,
        # keeping the order of samples
        final_response_examples = extra_ai_responses
        for resp in ai_response_examples:
            if resp not in final_response_examples:
                final_response_examples.append(resp)

        llm_response = await self.generate_answer(
            final_response_examples, conversation_samples, history
        )
        if not llm_response:
            structlogger.debug("intentless_policy.prediction.skip_llm_fail")
            return None, 0.0
        embedded_llm_response = self.embed_llm_response(llm_response)
        if not embedded_llm_response:
            structlogger.debug("intentless_policy.prediction.skip_response_embed_fail")
            return None, 0.0

        search_results = self.response_index.similarity_search_with_score_by_vector(
            embedded_llm_response
        )

        return self._get_response_and_score(search_results, tracker)

    def extract_ai_responses(self, conversation_samples: List[str]) -> List[str]:
        """Extracts the AI responses from the conversation samples.

        Args:
            conversation_samples: The conversation samples.

        Returns:
        The AI responses.
        """
        ai_replies = []

        for conversation_sample in conversation_samples:
            ai_texts = extract_participant_messages_from_transcript(
                conversation_sample, participant=AI
            )
            for ai_text in ai_texts:
                if ai_text not in ai_replies:
                    ai_replies.append(ai_text)
        return ai_replies

    def _get_response_and_score(
        self,
        search_results: List[Tuple["Document", float]],
        tracker: DialogueStateTracker,
    ) -> Tuple[Optional[str], float]:
        """Returns the response and score of the response.

        If there are no search results, returns None for the response
        and 0 for the score.

        Args:
            search_results: The search results.
            tracker: The tracker containing the conversation history up to now.


        Returns:
        The response and the score of the response.
        """
        if not search_results:
            return None, 0.0

        first_search_result = search_results[0]
        document, l2dist = first_search_result
        response = document.page_content
        score = 1.0 - float(l2dist) / math.sqrt(2)  # from L2 distance to score

        if tracker.latest_message:
            nlu_confidence = tracker.latest_message.intent.get(
                PREDICTED_CONFIDENCE_KEY, 1.0
            )
        else:
            nlu_confidence = 1.0
        # --- If we have high NLU confidence, let rules / memo run the show
        if nlu_confidence > self.nlu_abstention_threshold:
            score = min(score, nlu_confidence - 0.01)
        return response, score

    def select_response_examples(
        self,
        history: str,
        number_of_samples: int,
        max_number_of_tokens: int,
    ) -> List[str]:
        """Samples responses that fit the current conversation.

        Args:
            history: The conversation history.
            policy_model: The policy model.
            number_of_samples: The number of samples to return.
            max_number_of_tokens: Maximum number of tokens for responses.

        Returns:
            The sampled conversation in order of score decrease.
        """
        if not self.response_index or not history:
            return []

        try:
            embedding = self.embedder.embed_query(history)
        except Exception as e:
            structlogger.error(
                "intentless_policy.select_response_examples.error", error=e
            )
            return []

        docs = self.response_index.similarity_search_by_vector(
            embedding, k=number_of_samples
        )
        structlogger.debug(
            "intentless_policy.select_response_examples.success",
            number_of_samples=number_of_samples,
            docs=docs,
        )

        return [
            document.page_content
            for document in truncate_documents(
                docs, max_number_of_tokens=max_number_of_tokens
            )
        ]

    def select_few_shot_conversations(
        self,
        history: str,
        number_of_samples: int,
        max_number_of_tokens: int,
    ) -> List[str]:
        """Samples conversations from the given conversation samples.

        Excludes conversations without AI replies

        Args:
            history: The conversation history.
            number_of_samples: The number of samples to return.
            max_number_of_tokens: Maximum number of tokens for conversations.

        Returns:
            The sampled conversation ordered by similarity decrease.
        """
        if not self.conversation_samples_index or not history:
            return []

        docs = self.conversation_samples_index.similarity_search(
            history, k=number_of_samples
        )

        structlogger.debug(
            "intentless_policy.sampled_conversation",
            number_of_samples=number_of_samples,
            docs=docs,
        )
        conversations = []

        for doc in truncate_documents(docs, max_number_of_tokens=max_number_of_tokens):
            conversations.append(doc.page_content)
        return conversations

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
    ) -> "IntentlessPolicy":
        """Loads a trained policy (see parent class for full docstring)."""
        responses_docsearch = None
        samples_docsearch = None
        prompt_template = None
        try:
            with model_storage.read_from(resource) as path:
                responses_docsearch = load_faiss_vector_store(
                    path / "responses_faiss", cls._create_plain_embedder(config)
                )
                samples_docsearch = load_faiss_vector_store(
                    path / "samples_faiss", cls._create_plain_embedder(config)
                )

                # FIXME: This is a hack to make sure that the docsearches are
                #  normalized. unfortunatley langchain doesn't persist / load
                #  this parameter.
                if responses_docsearch:
                    responses_docsearch._normalize_L2 = True  # pylint: disable=protected-access
                prompt_template = rasa.shared.utils.io.read_file(
                    path / INTENTLESS_PROMPT_TEMPLATE_FILE_NAME
                )

        except (ValueError, FileNotFoundError, FileIOException) as e:
            structlogger.warning(
                "intentless_policy.load.failed", error=e, resource_name=resource.name
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            responses_docsearch=responses_docsearch,
            samples_docsearch=samples_docsearch,
            prompt_template=prompt_template,
        )

    @classmethod
    def fingerprint_addon(cls, config: Dict[str, Any]) -> Optional[str]:
        """Add a fingerprint of the knowledge base for the graph."""
        prompt_template = get_prompt_template(
            config.get(PROMPT_CONFIG_KEY),
            DEFAULT_INTENTLESS_PROMPT_TEMPLATE,
        )
        return deep_container_fingerprint(prompt_template)
