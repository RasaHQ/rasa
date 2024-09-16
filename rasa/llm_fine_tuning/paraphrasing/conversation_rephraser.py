import importlib.resources
import re
from typing import Dict, Any, List, Tuple, Optional

import structlog
from jinja2 import Template

from rasa.llm_fine_tuning.conversations import Conversation
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.constants import (
    MODEL_NAME_CONFIG_KEY,
    MODEL_CONFIG_KEY,
    LLM_CONFIG_KEY,
    PROMPT_TEMPLATE_CONFIG_KEY,
    PROVIDER_CONFIG_KEY,
    TIMEOUT_CONFIG_KEY,
)
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.providers.mappings import OPENAI_PROVIDER
from rasa.shared.utils.llm import (
    get_prompt_template,
    llm_factory,
    USER,
)

SEPARATOR = "\n\n"
BACKUP_SEPARATOR = "\nUSER:"

REPHRASING_PROMPT_FILE_NAME = "default_rephrase_prompt_template.jina2"
DEFAULT_REPHRASING_PROMPT_TEMPLATE = importlib.resources.read_text(
    "rasa.llm_fine_tuning.paraphrasing",
    REPHRASING_PROMPT_FILE_NAME,
)

DEFAULT_LLM_CONFIG = {
    PROVIDER_CONFIG_KEY: OPENAI_PROVIDER,
    MODEL_CONFIG_KEY: "gpt-4o-mini",
    TIMEOUT_CONFIG_KEY: 7,
    "temperature": 0.0,
    "max_tokens": 4096,
}

structlogger = structlog.get_logger()


class ConversationRephraser:
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.config = {**self.get_default_config(), **config}
        self.prompt_template = get_prompt_template(
            self.config.get(PROMPT_TEMPLATE_CONFIG_KEY),
            DEFAULT_REPHRASING_PROMPT_TEMPLATE,
        )

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """Validate the rephrase_config."""
        if LLM_CONFIG_KEY in config:
            llm_config = config.get(LLM_CONFIG_KEY)

            # Check if LLM configuration is set to None, {}.
            if llm_config is None:
                error = "LLM config is empty. Please provide a valid LLM config."
                structlogger.error("rephrase_config.empty_llm_config", error=error)
                raise ValueError(error)

            # Validate LLM model name or model in config.
            if not llm_config.get(MODEL_CONFIG_KEY) and not llm_config.get(
                MODEL_NAME_CONFIG_KEY
            ):
                error = (
                    "LLM model name is empty. Please provide a valid LLM model name."
                )
                structlogger.error("rephrase_config.llm_model_is_not_set", error=error)
                raise ValueError(error)

        # Check if the config contains only the allowed keys.
        allowed_keys = {PROMPT_TEMPLATE_CONFIG_KEY, LLM_CONFIG_KEY}
        if len(set(config.keys()) - allowed_keys) > 0:
            error = (
                f"Invalid rephrase config. Only the following keys are allowed: "
                f"{', '.join(allowed_keys)}."
            )
            structlogger.error("rephrase_config.invalid_keys", error=error)
            raise ValueError(error)

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            PROMPT_TEMPLATE_CONFIG_KEY: None,
            LLM_CONFIG_KEY: DEFAULT_LLM_CONFIG,
        }

    async def rephrase_conversation(
        self, conversation: Conversation, number_of_rephrasings: int = 10
    ) -> List[RephrasedUserMessage]:
        """Create rephrasings for each user message in the conversation.

        For each user message create <number_of_rephrasings> number of rephrasings.
        The rephrasings are created with an LLM.

        Args:
            conversation: The conversation.
            number_of_rephrasings: The number of rephrasings to produce per user
            message.

        Returns:
            A list of rephrased user messages.
        """
        prompt = self._render_template(conversation, number_of_rephrasings)

        result = await self._invoke_llm(prompt)

        rephrased_user_messages = self._parse_output(
            result, conversation.get_user_messages()
        )

        # Check if user message was successfully rephrased
        self._check_rephrasings(
            rephrased_user_messages, number_of_rephrasings, result, conversation.name
        )

        return rephrased_user_messages

    async def _invoke_llm(self, prompt: str) -> str:
        llm = llm_factory(self.config.get(LLM_CONFIG_KEY), DEFAULT_LLM_CONFIG)
        try:
            llm_response = await llm.acompletion(prompt)
            return llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error(
                "conversation_rephraser.rephrase_conversation.llm.error", error=e
            )
            raise ProviderClientAPIException(e, message="LLM call exception")

    def _render_template(
        self,
        conversation: Conversation,
        number_of_rephrasings: int,
    ) -> str:
        user_messages = conversation.get_user_messages()
        number_of_user_messages = len(user_messages)

        return Template(self.prompt_template).render(
            test_case_name=conversation.name,
            transcript=conversation.transcript,
            number_of_user_messages=number_of_user_messages,
            number_of_rephrasings=number_of_rephrasings,
            user_prefix=USER,
            user_messages=user_messages,
        )

    @staticmethod
    def _extract_rephrasings(block: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Extract the rephrasings for a specific user message.

        Expected format looks like this:
        USER: <user message>
        1. <rephrased message 1>
        2. <rephrased message 2>
        3. <rephrased message 3>
        ...

        Args:
            block: String that contains the user message and its rephrasings.

        Returns:
            The original user message and the list of rephrasings.
        """
        if not block.strip():
            return None, None

        # Split the block by new line character
        lines = block.strip().split("\n")
        # Filter out empty lines and lines that are equal to """
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line and line != '"""']

        # We need at least the original user message and one rephrasing
        if len(lines) < 2:
            return None, None

        # Extract the user message from the first line
        # (ideally prefixed with 'USER: ')
        if lines[0].startswith("USER:"):
            original_user_message = lines[0][len(f"{USER}:") :].strip()
        else:
            original_user_message = lines[0]

        # Extract rephrasings
        rephrasings = []
        for line in lines[1:]:
            # Remove bullets or numbering and any extra whitespace
            line = re.sub(r"^\s*[-\d\.]+", "", line).strip()
            if line.startswith("USER:"):
                line = line[len(f"{USER}:") :].strip()
            if line:
                rephrasings.append(line)

        return original_user_message, rephrasings

    def _parse_output(
        self, output: str, user_messages: List[str]
    ) -> List[RephrasedUserMessage]:
        rephrased_messages = [
            RephrasedUserMessage(message, []) for message in user_messages
        ]

        message_blocks = self._get_message_blocks(output, len(rephrased_messages))
        for block in message_blocks:
            original_user_message, rephrasings = self._extract_rephrasings(block)

            if not original_user_message or not rephrasings:
                continue

            # Add the rephrasings to the correct user message
            for rephrased_message in rephrased_messages:
                if (
                    rephrased_message.original_user_message.lower()
                    == original_user_message.lower()
                ):
                    rephrased_message.rephrasings = rephrasings

        return rephrased_messages

    @staticmethod
    def _get_message_blocks(output: str, expected_number_of_blocks: int) -> List[str]:
        # Each user message block is (ideally) separated by new line ("\n\n")
        # USER: <user message>
        # 1. <rephrasing>
        #
        # USER: <user message>
        # 1. <rephrasing>
        message_blocks = output.split(SEPARATOR)
        if len(message_blocks) == expected_number_of_blocks:
            return message_blocks

        # In case the message blocks are not separated by new line, try to split it
        # by "\nUSER:", e.g.
        # USER: <user message>
        # 1. <rephrasing>
        # USER: <user message>
        # 1. <rephrasing>
        message_blocks = output.split(BACKUP_SEPARATOR)
        if len(message_blocks) == expected_number_of_blocks:
            return message_blocks

        return []

    @staticmethod
    def _check_rephrasings(
        rephrased_messages: List[RephrasedUserMessage],
        number_of_rephrasings: int,
        llm_output: str,
        conversation_name: str,
    ) -> None:
        incorrect_rephrasings_for_messages = []

        for message in rephrased_messages:
            if (
                not message.rephrasings
                or len(message.rephrasings) != number_of_rephrasings
            ):
                incorrect_rephrasings_for_messages.append(message.original_user_message)

        if incorrect_rephrasings_for_messages:
            structlogger.warning(
                "conversation_rephraser.rephrase_conversation.parse_llm_output",
                warning="Failed to parse llm output correctly. Not all user messages"
                "were successfully rephrased.",
                llm_output=llm_output,
                conversation_name=conversation_name,
                incorrect_rephrasings_for_messages=incorrect_rephrasings_for_messages,
            )
