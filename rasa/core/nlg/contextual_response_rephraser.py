from typing import Any, Dict, Optional, Text

import structlog
from jinja2 import Template

from rasa import telemetry
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.shared.constants import (
    LLM_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    MODEL_NAME_CONFIG_KEY,
    PROMPT_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    OPENAI_PROVIDER,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.core.domain import KEY_RESPONSES_TEXT, Domain
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import (
    DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    USER,
    combine_custom_and_default_config,
    get_prompt_template,
    llm_factory,
    try_instantiate_llm_client,
)
from rasa.utils.endpoints import EndpointConfig

from rasa.core.nlg.summarize import summarize_conversation

from rasa.utils.log_utils import log_llm

structlogger = structlog.get_logger()

RESPONSE_REPHRASING_KEY = "rephrase"

RESPONSE_REPHRASING_TEMPLATE_KEY = "rephrase_prompt"

DEFAULT_REPHRASE_ALL = False

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: DEFAULT_OPENAI_GENERATE_MODEL_NAME,
    "temperature": 0.3,
    "max_tokens": DEFAULT_OPENAI_MAX_GENERATED_TOKENS,
    TIMEOUT_CONFIG_KEY: 5,
}

DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE = """The following is a conversation with
an AI assistant. The assistant is helpful, creative, clever, and very friendly.
Rephrase the suggested AI response staying close to the original message and retaining
its meaning. Use simple english.

Context / previous conversation with the user:
{{history}}

{{current_input}}

Suggested AI Response: {{suggested_response}}

Rephrased AI Response:"""


class ContextualResponseRephraser(TemplatedNaturalLanguageGenerator):
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
        self.prompt_template = get_prompt_template(
            self.nlg_endpoint.kwargs.get(PROMPT_CONFIG_KEY),
            DEFAULT_RESPONSE_VARIATION_PROMPT_TEMPLATE,
        )
        self.rephrase_all = self.nlg_endpoint.kwargs.get(
            "rephrase_all", DEFAULT_REPHRASE_ALL
        )
        self.trace_prompt_tokens = self.nlg_endpoint.kwargs.get(
            "trace_prompt_tokens", False
        )
        try_instantiate_llm_client(
            self.nlg_endpoint.kwargs.get(LLM_CONFIG_KEY),
            DEFAULT_LLM_CONFIG,
            "contextual_response_rephraser.init",
            "ContextualResponseRephraser",
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

    async def _generate_llm_response(self, prompt: str) -> Optional[str]:
        """Use LLM to generate a response.

        Args:
            prompt: the prompt to send to the LLM

        Returns:
            generated text
        """
        llm = llm_factory(
            self.nlg_endpoint.kwargs.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
        )

        try:
            llm_response = await llm.acompletion(prompt)
            return llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error("nlg.llm.error", error=e)
            return None

    def llm_property(self, prop: str) -> Optional[str]:
        """Returns a property of the LLM provider."""
        return combine_custom_and_default_config(
            self.nlg_endpoint.kwargs.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
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
        llm = llm_factory(
            self.nlg_endpoint.kwargs.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG
        )
        return await summarize_conversation(tracker, llm, max_turns=5)

    async def rephrase(
        self,
        response: Dict[str, Any],
        tracker: DialogueStateTracker,
    ) -> Dict[str, Any]:
        """Predicts a variation of the response.

        Args:
            response: The response to rephrase.
            tracker: The tracker to use for the prediction.
            model_name: The name of the model to use for the prediction.

        Returns:
            The response with the rephrased text.
        """
        if not (response_text := response.get(KEY_RESPONSES_TEXT)):
            return response

        latest_message = self._last_message_if_human(tracker)
        current_input = f"{USER}: {latest_message}" if latest_message else ""

        prompt_template_text = self._template_for_response_rephrasing(response)

        prompt = Template(prompt_template_text).render(
            history=await self._create_history(tracker),
            suggested_response=response_text,
            current_input=current_input,
            slots=tracker.current_slot_values(),
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
        )
        if not (updated_text := await self._generate_llm_response(prompt)):
            # If the LLM fails to generate a response, we
            # return the original response.
            return response

        structlogger.debug(
            "nlg.rewrite.complete",
            response_text=response_text,
            updated_text=updated_text,
        )
        response[KEY_RESPONSES_TEXT] = updated_text
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
        output_channel: Text,
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
        filled_slots = tracker.current_slot_values()
        stack_context = tracker.stack.current_context()
        templated_response = self.generate_from_slots(
            utter_action=utter_action,
            filled_slots=filled_slots,
            stack_context=stack_context,
            output_channel=output_channel,
            **kwargs,
        )

        if templated_response and self.does_response_allow_rephrasing(
            templated_response
        ):
            return await self.rephrase(
                templated_response,
                tracker,
            )
        else:
            return templated_response
