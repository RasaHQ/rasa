from typing import List, Optional

import structlog

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.commands import Command, SetSlotCommand
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.utils.llm import (
    create_tracker_for_user_step,
    generate_sender_id,
)

structlogger = structlog.get_logger()


class RephraseValidator:
    def __init__(self, flows: FlowsList) -> None:
        self.flows = flows

    async def validate_rephrasings(
        self,
        agent: Agent,
        rephrasings: List[RephrasedUserMessage],
        conversation: Conversation,
    ) -> List[RephrasedUserMessage]:
        """Split rephrased user messages into passing and failing.

        Handle the rephrased messages using agent the same way the original
        message was handled. Check if the rephrased user message is producing
        the same commands as the original user message. The rephrase is passing
        if the commands match and failing otherwise.

        Args:
            agent: Rasa agent
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
                if await self._validate_rephrase_is_passing(
                    agent,
                    rephrase,
                    step,
                    conversation.name,
                    conversation.tracker,
                ):
                    current_rephrasings.passed_rephrasings.append(rephrase)
                else:
                    current_rephrasings.failed_rephrasings.append(rephrase)

        return rephrasings

    async def _validate_rephrase_is_passing(
        self,
        agent: Agent,
        rephrase: str,
        step: ConversationStep,
        test_case_name: str,
        tracker: DialogueStateTracker,
    ) -> bool:
        rephrased_tracker = await self._send_rephrased_message_to_agent(
            rephrase, step, test_case_name, agent, tracker
        )
        if not (rephrased_tracker and rephrased_tracker.latest_message):
            return False

        commands_from_original_utterance = step.llm_commands

        commands_from_rephrased_utterance = [
            Command.command_from_json(command_json)
            for command_json in rephrased_tracker.latest_message.commands
        ]

        return self._check_commands_match(
            commands_from_original_utterance, commands_from_rephrased_utterance
        )

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
    async def _send_rephrased_message_to_agent(
        rephrased_user_message: str,
        step: ConversationStep,
        test_case_name: str,
        agent: Agent,
        tracker: DialogueStateTracker,
    ) -> Optional[DialogueStateTracker]:
        # create a rephrased UserMessage
        sender_id = generate_sender_id(test_case_name)
        user_message = UserMessage(rephrased_user_message, sender_id=sender_id)

        await create_tracker_for_user_step(
            sender_id, agent, tracker, step.tracker_event_index
        )

        await agent.handle_message(user_message)
        rephrased_tracker = await agent.tracker_store.retrieve(sender_id)

        return rephrased_tracker
