from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import structlog
from tqdm import tqdm

from rasa.core.agent import Agent
from rasa.core.channels import UserMessage
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.utils import set_record_commands_and_prompts
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.storage import StorageContext
from rasa.llm_fine_tuning.utils import (
    commands_as_string,
    make_mock_invoke_llm,
    patch_invoke_llm_in_generators,
)
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import KEY_USER_PROMPT, PROMPTS
from rasa.shared.utils.llm import generate_sender_id

LLM_DATA_PREPARATION_MODULE_STORAGE_LOCATION = "3_llm_finetune_data/llm_ft_data.jsonl"

structlogger = structlog.get_logger()


@dataclass
class LLMDataExample:
    prompt: str
    output: List[PromptCommand]
    original_test_name: str
    original_user_utterance: str
    rephrased_user_utterance: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "output": commands_as_string(self.output),
            "original_test_name": self.original_test_name,
            "original_user_utterance": self.original_user_utterance,
            "rephrased_user_utterance": self.rephrased_user_utterance,
        }


def _create_data_point(
    prompt: str,
    step: ConversationStep,
    conversation: Conversation,
    rephrased_user_message: Optional[str] = None,
) -> LLMDataExample:
    return LLMDataExample(
        prompt,
        step.llm_commands,
        conversation.get_full_name(),
        step.original_test_step.text,
        rephrased_user_message,
    )


async def _convert_conversation_into_llm_data(
    conversation: Conversation, agent: Agent
) -> List[LLMDataExample]:
    data = []

    # construct new conversations from the rephrasings
    new_conversations = _construct_new_conversations(conversation)

    original_user_steps = [
        step for step in conversation.iterate_over_annotated_user_steps()
    ]

    for i, step in enumerate(original_user_steps):
        # create data point for the original e2e test case
        data.append(_create_data_point(step.llm_prompt, step, conversation))

    test_case_name = conversation.name

    # create data points using the rephrasings, e.g. 'new_conversations'
    for rephrased_user_steps in new_conversations:
        sender_id = generate_sender_id(test_case_name)
        # create a new tracker to be able to simulate the conversation from start
        await agent.tracker_store.save(DialogueStateTracker(sender_id, slots=[]))
        # simulate the conversation to get the prompts
        for i, step in enumerate(original_user_steps):
            rephrased_user_message = rephrased_user_steps[i]
            user_message = UserMessage(rephrased_user_message, sender_id=sender_id)

            expected_commands = "\n".join(
                [command.to_dsl() for command in step.llm_commands]
            )
            fake_invoke_function = make_mock_invoke_llm(expected_commands)

            with (
                set_record_commands_and_prompts(),
                patch_invoke_llm_in_generators(fake_invoke_function),
            ):
                await agent.handle_message(user_message)

            rephrased_tracker = await agent.tracker_store.retrieve(sender_id)
            if rephrased_tracker is None:
                # if tracker doesn't exist, we can't create a data point
                continue

            latest_message = rephrased_tracker.latest_message
            if latest_message is None:
                # if there is no latest message, we don't create a data point
                continue

            # tell the type checker what we expect to find under "prompts"
            prompts = cast(
                Optional[List[Dict[str, Any]]], latest_message.parse_data.get(PROMPTS)
            )

            if prompts:
                # as we only use single step or compact command generator,
                # there is always exactly one prompt
                prompt = prompts[0]
                user_prompt: Optional[str] = prompt.get(KEY_USER_PROMPT)
                data.append(
                    _create_data_point(
                        user_prompt, step, conversation, rephrased_user_message
                    )
                )

    return data


def _construct_new_conversations(conversation: Conversation) -> List[List[str]]:
    """Construct new conversations from the rephrasings.

    In general, we will combine the passing rephrasings at the same index position to
    construct a new conversation. If for one particular user turn no other passing
    rephrasing exists, we reset the index and take the first passing rephrasing again.

    Args:
        conversation: The conversation.

    Returns:
        A list of new conversations (only rephrased user turns).
    """
    max_passed_rephrasings = max(
        [
            len(step.passed_rephrasings)
            for step in conversation.iterate_over_annotated_user_steps()
        ]
    )

    new_conversations = []
    for i in range(0, max_passed_rephrasings):
        current_conversation = []
        for step in conversation.iterate_over_annotated_user_steps():
            # take the orginial user message in case no passing rephrasings exist
            if not step.passed_rephrasings and step.original_test_step.text:
                structlogger.debug(
                    "llm_fine_tuning.llm_data_preparation_module."
                    "construct_new_conversations.no_passed_rephrasings",
                    conversation=conversation.get_full_name(),
                    step=step.original_test_step.text,
                    message="Take original user message instead of rephrasing.",
                )
                current_conversation.append(step.original_test_step.text)
                continue

            # some user steps might have fewer rephrasings than others
            # loop over the rephrasings
            index = i % len(step.passed_rephrasings)
            current_conversation.append(step.passed_rephrasings[index])
        if current_conversation:
            new_conversations.append(current_conversation)

    structlogger.debug(
        "llm_fine_tuning.llm_data_preparation_module.construct_new_conversations",
        conversation=conversation.get_full_name(),
        new_conversations=new_conversations,
    )

    return new_conversations


async def convert_to_fine_tuning_data(
    conversations: List[Conversation],
    storage_context: StorageContext,
    agent: Agent,
) -> List[LLMDataExample]:
    llm_data = []

    for i in tqdm(range(len(conversations))):
        conversation_llm_data = await _convert_conversation_into_llm_data(
            conversations[i], agent
        )
        llm_data.extend(conversation_llm_data)

    storage_context.write_llm_data(
        llm_data, LLM_DATA_PREPARATION_MODULE_STORAGE_LOCATION
    )

    return llm_data
