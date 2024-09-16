from typing import Dict, Any, List

import structlog

from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.dialogue_understanding.generator import SingleStepLLMCommandGenerator
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.utils.llm import (
    llm_factory,
)

structlogger = structlog.get_logger()


class RephraseValidator:
    def __init__(self, llm_config: Dict[str, Any], flows: FlowsList) -> None:
        self.llm_config = llm_config
        self.flows = flows

    async def validate_rephrasings(
        self,
        rephrasings: List[RephrasedUserMessage],
        conversation: Conversation,
    ) -> List[RephrasedUserMessage]:
        """Split rephrased user messages into passing and failing.

        Call an LLM using the same config of the former trained model with an updated
        prompt from the original user message (replace all occurrences of the original
        user message with the rephrased user message). Check if the
        rephrased user message is producing the same commands as the original user
        message. The rephase is passing if the commands match and failing otherwise.

        Args:
            rephrasings: The rephrased user messages.
            conversation: The conversation.

        Returns:
            A list of rephrased user messages including the passing and failing
            rephrases.
        """
        for i, step in enumerate(
            conversation.iterate_over_annotated_user_steps(rephrase=True)
        ):
            current_rephrasings = rephrasings[i]

            for rephrase in current_rephrasings.rephrasings:
                if await self._validate_rephrase_is_passing(rephrase, step):
                    current_rephrasings.passed_rephrasings.append(rephrase)
                else:
                    current_rephrasings.failed_rephrasings.append(rephrase)

        return rephrasings

    async def _validate_rephrase_is_passing(
        self,
        rephrase: str,
        step: ConversationStep,
    ) -> bool:
        prompt = self._update_prompt(
            rephrase, step.original_test_step.text, step.llm_prompt
        )

        action_list = await self._invoke_llm(prompt)

        commands_from_original_utterance = step.llm_commands
        commands_from_rephrased_utterance = (
            SingleStepLLMCommandGenerator.parse_commands(action_list, None, self.flows)
        )
        return self._check_commands_match(
            commands_from_original_utterance, commands_from_rephrased_utterance
        )

    async def _invoke_llm(self, prompt: str) -> str:
        from rasa.dialogue_understanding.generator.constants import DEFAULT_LLM_CONFIG

        llm = llm_factory(self.llm_config, DEFAULT_LLM_CONFIG)

        try:
            llm_response = await llm.acompletion(prompt)
            return llm_response.choices[0]
        except Exception as e:
            # unfortunately, langchain does not wrap LLM exceptions which means
            # we have to catch all exceptions here
            structlogger.error(
                "rephrase_validator.validate_conversation.llm.error", error=e
            )
            raise ProviderClientAPIException(e, message="LLM call exception")

    @staticmethod
    def _check_commands_match(
        expected_commands: List[Command], actual_commands: List[Command]
    ) -> bool:
        if len(expected_commands) != len(actual_commands):
            return False

        for expected_command in expected_commands:
            if isinstance(expected_command, SetSlotCommand):
                slot_name = expected_command.name
                match_found = False
                for c in actual_commands:
                    if isinstance(c, SetSlotCommand) and c.name == slot_name:
                        match_found = True
                        break
                if not match_found:
                    return False

            elif expected_command not in actual_commands:
                return False

        return True

    @staticmethod
    def _update_prompt(
        rephrased_user_message: str, original_user_message: str, prompt: str
    ) -> str:
        return prompt.replace(original_user_message, rephrased_user_message)
