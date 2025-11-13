from typing import Any, Dict, List, Optional

import structlog

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.generator.constants import (
    DEFAULT_LLM_CONFIG,
)
from rasa.shared.constants import TEXT
from rasa.shared.core.constants import (
    ACTION_LLM_STREAMING_RESPONSE,
    ACTION_METADATA_LLM_CONFIG_KEY,
    ACTION_METADATA_MESSAGE_KEY,
    ACTION_METADATA_PROMPT_KEY,
    ACTION_METADATA_TEXT_KEY,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import acompletion_with_streaming, llm_factory

logger = structlog.get_logger()


class ActionLLMStreamingResponse(Action):
    """Action which handles streaming responses from the LLM.

    This action takes a prompt and an LLM configuration from the action metadata.
    It uses the helper function `acompletion_with_streaming` to stream the response
    as it's being generated to the user via the output channel.
    Used by Enterprise Search Policy.
    """

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_LLM_STREAMING_RESPONSE

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Execute the action to stream LLM response.

        Args:
            output_channel: Channel to send the streaming response to.
            nlg: Natural language generator (not used in this action).
            tracker: Current dialogue state tracker.
            domain: Domain object.
            metadata: Action metadata containing llm_config and prompt.

        Returns:
            Bot utterance event with the generated LLM response.
        """
        if not metadata:
            logger.error("action_llm_streaming.run.no_metadata")
            return []

        llm_config = metadata.get(ACTION_METADATA_LLM_CONFIG_KEY)
        prompt = metadata.get(ACTION_METADATA_PROMPT_KEY)
        action_metadata = metadata.get(ACTION_METADATA_MESSAGE_KEY, {})

        if not llm_config or not prompt:
            logger.error(
                "action_llm_streaming.run.missing_required_metadata",
                llm_config_present=bool(llm_config),
                prompt_present=bool(prompt),
            )
            return []

        try:
            llm = llm_factory(llm_config, DEFAULT_LLM_CONFIG)
            response = await acompletion_with_streaming(
                llm_client=llm,
                messages=prompt,
                output_channel=output_channel,
                recipient_id=tracker.sender_id,
            )
        except Exception as e:
            logger.error(
                "action_llm_streaming.run.error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        if not response or not response.choices:
            logger.error("action_llm_streaming.run.no_response")
            return []

        llm_answer = response.choices[0]
        action_metadata[TEXT] = llm_answer
        action_metadata[ACTION_METADATA_MESSAGE_KEY][ACTION_METADATA_TEXT_KEY] = (
            llm_answer
        )
        return [create_bot_utterance(action_metadata)]
