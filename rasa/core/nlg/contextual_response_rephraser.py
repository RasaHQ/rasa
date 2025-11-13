from typing import Any, Dict, Optional, Text

import structlog
from jinja2 import Template

from rasa import telemetry
from rasa.core.channels import OutputChannel
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.core.nlg.summarize import (
    _count_multiple_utterances_as_single_turn,
    summarize_conversation,
)
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    MAX_COMPLETION_TOKENS_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_GROUP_ID_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    OPENAI_PROVIDER,
    PROMPT_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    TEMPERATURE_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.domain import KEY_RESPONSES_TEXT, Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.flows.constants import KEY_TRANSLATION
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import (
    KEY_COMPONENT_NAME,
    KEY_LLM_RESPONSE_METADATA,
    KEY_PROMPT_NAME,
    KEY_USER_PROMPT,
    PROMPTS,
)
from rasa.shared.providers.llm.llm_response import LLMResponse
from rasa.shared.utils.constants import (
    LANGFUSE_METADATA_AGENT_ID,
    LANGFUSE_METADATA_COMPONENT_NAME,
    LANGFUSE_METADATA_CUSTOM_METADATA,
    LANGFUSE_METADATA_MODEL_ID,
    LANGFUSE_METADATA_SESSION_ID,
    LANGFUSE_METADATA_TAGS,
    LOG_COMPONENT_SOURCE_METHOD_INIT,
)
from rasa.shared.utils.health_check.llm_health_check_mixin import LLMHealthCheckMixin
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    USER,
    LLMInput,
    acompletion_with_streaming,
    check_prompt_config_keys_and_warn_if_deprecated,
    combine_custom_and_default_config,
    get_prompt_template,
    llm_factory,
    resolve_model_client_config,
    tracker_as_readable_transcript,
)
from rasa.utils.endpoints import EndpointConfig
from rasa.utils.log_utils import log_llm

structlogger = structlog.get_logger()

RESPONSE_REPHRASING_KEY = "rephrase"

RESPONSE_REPHRASING_TEMPLATE_KEY = "rephrase_prompt"

RESPONSE_SUMMARISE_CONVERSATION_KEY = "summarize_conversation"

DEFAULT_REPHRASE_ALL = False
DEFAULT_SUMMARIZE_HISTORY = True
DEFAULT_MAX_HISTORICAL_TURNS = 5
DEFAULT_COUNT_MULTIPLE_UTTERANCES_AS_SINGLE_TURN = True

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    TEMPERATURE_CONFIG_KEY: 0.3,
    MAX_COMPLETION_TOKENS_CONFIG_KEY: DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 5,
}

DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE = """The following is a conversation with
an AI assistant. The assistant is helpful, creative, clever, and very friendly.
Rephrase the suggested AI response staying close to the original message and retaining
its meaning. Use simple {{language}}.

Context / previous conversation with the user:
{{history}}

Last user message:
{{current_input}}

Suggested AI Response: {{suggested_response}}

Rephrased AI Response:"""


class ContextualResponseRephraser(
    LLMHealthCheckMixin, TemplatedNaturalLanguageGenerator
):
    """Generates responses based on modified templates.

    The templates are filled with the entities and slots that are available in the
    tracker. The resulting response is then passed through the LLM to generate a
    variation of the response.

    The variation is only generated if the response is a text response and the
    response explicitly set `rephrase` to `True`. This is to avoid
    generating responses for templates that are to volatile to be
    modified by the LLM.

    Args:
        endpoint_config: The endpoint configuration for the LLM.
        domain: The domain of the assistant.

    Attributes:
        nlg_endpoint: The endpoint configuration for the LLM.
    """

    def __init__(self, endpoint_config: EndpointConfig, domain: Domain) -> None:
        super().__init__(domain.responses)

        self.nlg_endpoint = endpoint_config

        # Warn if the prompt config key is used to set the prompt template
        check_prompt_config_keys_and_warn_if_deprecated(
            self.nlg_endpoint.kwargs, "contextual_response_rephraser"
        )

        self.prompt_template = get_prompt_template(
            self.nlg_endpoint.kwargs.get(PROMPT_TEMPLATE_CONFIG_KEY)
            or self.nlg_endpoint.kwargs.get(PROMPT_CONFIG_KEY),
            DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE,
            log_source_component=ContextualResponseRephraser.__name__,
            log_source_method=LOG_COMPONENT_SOURCE_METHOD_INIT,
        )
        self.rephrase_all = self.nlg_endpoint.kwargs.get(
            "rephrase_all", DEFAULT_REPHRASE_ALL
        )
        self.trace_prompt_tokens = self.nlg_endpoint.kwargs.get(
            "trace_prompt_tokens", False
        )
        self.summarize_history = self.nlg_endpoint.kwargs.get(
            "summarize_history", DEFAULT_SUMMARIZE_HISTORY
        )
        self.max_historical_turns = self.nlg_endpoint.kwargs.get(
            "max_historical_turns", DEFAULT_MAX_HISTORICAL_TURNS
        )

        self.count_multiple_utterances_as_single_turn = self.nlg_endpoint.kwargs.get(
            "count_multiple_utterances_as_single_turn",
            DEFAULT_COUNT_MULTIPLE_UTTERANCES_AS_SINGLE_TURN,
        )

        self.llm_config = resolve_model_client_config(
            self.nlg_endpoint.kwargs.get(LLM_CONFIG_KEY),
            ContextualResponseRephraser.__name__,
        )

        self.perform_llm_health_check(
            self.llm_config,
            DEFAULT_LLM_CONFIG,
            "contextual_response_rephraser.init",
            ContextualResponseRephraser.__name__,
        )

    @classmethod
    def _add_prompt_and_llm_metadata_to_response(
        cls,
        response: Dict[str, Any],
        prompt_name: str,
        user_prompt: str,
        llm_response: Optional["LLMResponse"] = None,
    ) -> Dict[str, Any]:
        """Stores the prompt and LLMResponse metadata to response.

        Args:
            response: The response to add the prompt and LLMResponse metadata to.
            prompt_name: A name identifying prompt usage.
            user_prompt: The user prompt that was sent to the LLM.
            llm_response: The response object from the LLM (None if no response).
        """
        from rasa.dialogue_understanding.utils import record_commands_and_prompts

        if not record_commands_and_prompts:
            return response

        prompt_data: Dict[Text, Any] = {
            KEY_COMPONENT_NAME: cls.__name__,
            KEY_PROMPT_NAME: prompt_name,
            KEY_USER_PROMPT: user_prompt,
            KEY_LLM_RESPONSE_METADATA: llm_response.to_dict() if llm_response else None,
        }

        prompts = response.get(PROMPTS, [])
        prompts.append(prompt_data)
        response[PROMPTS] = prompts
        return response

    @staticmethod
    def get_language_label(tracker: DialogueStateTracker) -> str:
        """Fetches the label of the language to be used for the rephraser.

        Args:
            tracker: The tracker to get the language from.

        Returns:
            The label of the current language, or "English" if no language is set.
        """
        return (
            tracker.current_language.label
            if tracker.current_language
            else tracker.default_language.label
        )

    def _last_message_if_human(self, tracker: DialogueStateTracker) -> Optional[str]:
        """Returns the latest message from the tracker.

        If the latest message is from the AI, it returns None.

        Args:
            tracker: The tracker to get the latest message from.

        Returns:
            The latest message from the tracker if it is from the user, else None.
        """
        for event in reversed(tracker.events):
            if isinstance(event, UserUttered):
                return event.text
            if isinstance(event, BotUttered):
                return None
        return None

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

    async def _generate_llm_response(
        self,
        llm_input: LLMInput,
        output_channel: OutputChannel,
        recipient_id: str,
    ) -> Optional[LLMResponse]:
        """Generate LLM response with streaming support.

        Args:
            llm_input: LLMInput object with prompt and metadata.
            output_channel: Output channel to send streaming chunks to.
            recipient_id: Recipient ID for the output channel.

        Returns:
            The LLM response.
        """
        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)
        llm_response = await acompletion_with_streaming(
            llm_client=llm,
            messages=llm_input.prompt,
            metadata=llm_input.metadata,
            output_channel=output_channel,
            recipient_id=recipient_id,
        )
        return llm_response

    def llm_property(self, prop: str) -> Optional[str]:
        """Returns a property of the LLM provider."""
        return combine_custom_and_default_config(
            self.llm_config, DEFAULT_LLM_CONFIG
        ).get(prop)

    def custom_prompt_template(self, prompt_template: str) -> Optional[str]:
        """Returns the custom prompt template if it is not the default one."""
        if prompt_template != DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE:
            return prompt_template
        else:
            return None

    def _template_for_response_rephrasing(self, response: Dict[str, Any]) -> str:
        """Returns the template for the response rephrasing.

        Args:
            response: The response to rephrase.

        Returns:
            The template for the response rephrasing.
        """
        return response.get("metadata", {}).get(
            RESPONSE_REPHRASING_TEMPLATE_KEY, self.prompt_template
        )

    async def _create_history(self, tracker: DialogueStateTracker) -> str:
        """Creates the history for the prompt.

        Args:
            tracker: The tracker to use for the history.


        Returns:
        The history for the prompt.
        """
        # Count multiple utterances by bot/user as single turn in conversation history
        turns_wrapper = (
            _count_multiple_utterances_as_single_turn
            if self.count_multiple_utterances_as_single_turn
            else None
        )
        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)
        return await summarize_conversation(
            tracker, llm, max_turns=5, turns_wrapper=turns_wrapper
        )

    async def rephrase(
        self,
        response: Dict[str, Any],
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
    ) -> Dict[str, Any]:
        """Predicts a variation of the response.

        Args:
            response: The response to rephrase.
            tracker: The tracker to use for the prediction.
            model_name: The name of the model to use for the prediction.

        Returns:
            The response with the rephrased text.
        """
        translation_response = response.get(KEY_TRANSLATION) or {}
        lang_code = getattr(tracker.current_language, "code", None)
        response_text = translation_response.get(
            lang_code, response.get(KEY_RESPONSES_TEXT)
        )
        if not response_text:
            return response

        prompt_template_text = self._template_for_response_rephrasing(response)

        # Last user message (=current input) should always be in prompt if available
        last_message_by_user = getattr(tracker.latest_message, "text", "")
        current_input = (
            f"{USER}: {last_message_by_user}" if last_message_by_user else ""
        )

        # Only summarise conversation history if flagged
        if self.summarize_history:
            history = await self._create_history(tracker)
        else:
            # Count multiple utterances by bot/user as single turn
            turns_wrapper = (
                _count_multiple_utterances_as_single_turn
                if self.count_multiple_utterances_as_single_turn
                else None
            )
            max_turns = max(self.max_historical_turns, 1)
            history = tracker_as_readable_transcript(
                tracker, max_turns=max_turns, turns_wrapper=turns_wrapper
            )

        prompt = Template(prompt_template_text).render(
            history=history,
            suggested_response=response_text,
            current_input=current_input,
            slots=tracker.current_slot_values(),
            language=self.get_language_label(tracker),
        )
        log_llm(
            logger=structlogger,
            log_module="ContextualResponseRephraser",
            log_event="nlg.rephrase.prompt_rendered",
            prompt=prompt,
        )
        telemetry.track_response_rephrase(
            rephrase_all=self.rephrase_all,
            custom_prompt_template=self.custom_prompt_template(prompt_template_text),
            llm_type=self.llm_property(PROVIDER_CONFIG_KEY),
            llm_model=self.llm_property(MODEL_CONFIG_KEY)
            or self.llm_property(MODEL_NAME_CONFIG_KEY),
            llm_model_group_id=self.llm_property(MODEL_GROUP_ID_CONFIG_KEY),
        )
        llm_response = await self._generate_llm_response(
            LLMInput(prompt=prompt, metadata=self.get_llm_tracing_metadata(tracker)),
            output_channel=output_channel,
            recipient_id=tracker.sender_id,
        )
        llm_response = LLMResponse.ensure_llm_response(llm_response)

        response = self._add_prompt_and_llm_metadata_to_response(
            response=response,
            prompt_name="rephrase_prompt",
            user_prompt=prompt,
            llm_response=llm_response,
        )

        if not (llm_response and llm_response.choices and llm_response.choices[0]):
            # If the LLM fails to generate a response, return the original response.
            return response

        updated_text = llm_response.choices[0]

        if lang_code in translation_response:
            response[KEY_TRANSLATION][lang_code] = updated_text
        else:
            response[KEY_RESPONSES_TEXT] = updated_text

        structlogger.debug(
            "nlg.rewrite.complete",
            response_text=response_text,
            updated_text=updated_text,
        )
        return response

    def does_response_allow_rephrasing(self, template: Dict[Text, Any]) -> bool:
        """Checks if the template allows variation.

        Args:
            template: The template to check.

        Returns:
            `True` if the template allows variation, else `False`.
        """
        return template.get("metadata", {}).get(
            RESPONSE_REPHRASING_KEY, self.rephrase_all
        )

    async def generate(
        self,
        utter_action: Text,
        tracker: DialogueStateTracker,
        output_channel: OutputChannel,
        **kwargs: Any,
    ) -> Optional[Dict[Text, Any]]:
        """Generate a response for the requested utter action.

        Args:
            utter_action: The name of the utter action to generate a response for.
            tracker: The tracker to use for the generation.
            output_channel: The output channel to use for the generation.
            **kwargs: Additional arguments to pass to the generation.

        Returns:
            The generated response.
        """
        templated_response = await super().generate(
            utter_action=utter_action,
            tracker=tracker,
            output_channel=output_channel,
            **kwargs,
        )

        if templated_response and self.does_response_allow_rephrasing(
            templated_response
        ):
            return await self.rephrase(
                templated_response,
                tracker,
                output_channel,
            )
        else:
            return templated_response
