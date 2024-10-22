from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import structlog
from tqdm import tqdm

from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.storage import StorageContext

LLM_DATA_PREPARATION_MODULE_STORAGE_LOCATION = "3_llm_finetune_data/llm_ft_data.jsonl"

structlogger = structlog.get_logger()


@dataclass
class LLMDataExample:
    prompt: str
    output: str
    original_test_name: str
    original_user_utterance: str
    rephrased_user_utterance: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "output": self.output,
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
        step.commands_as_string(),
        conversation.get_full_name(),
        step.original_test_step.text,
        rephrased_user_message,
    )


def _update_prompt(
    prompt: str,
    original_user_steps: List[ConversationStep],
    rephrased_user_steps: List[str],
) -> Optional[str]:
    if len(original_user_steps) != len(rephrased_user_steps):
        structlogger.debug(
            "llm_fine_tuning.llm_data_preparation_module.failed_to_update_prompt",
            original_user_steps=[
                step.original_test_step.text for step in original_user_steps
            ],
            rephrased_user_steps=rephrased_user_steps,
        )
        return None

    updated_prompt = prompt
    for user_step, rephrased_message in zip(original_user_steps, rephrased_user_steps):
        # replace all occurrences of the original user message with the rephrased user
        # message in the conversation history mentioned in the prompt
        updated_prompt = updated_prompt.replace(
            f"USER: {user_step.original_test_step.text}", f"USER: {rephrased_message}"
        )

    # replace the latest user message mentioned in the prompt
    updated_prompt = updated_prompt.replace(
        f"'''{original_user_steps[-1].original_test_step.text}'''",
        f"'''{rephrased_user_steps[-1]}'''",
    )

    return updated_prompt


def _convert_conversation_into_llm_data(
    conversation: Conversation,
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

        # create data points using the rephrasings, e.g. 'new_conversations'
        for rephrased_user_steps in new_conversations:
            # +1 to include the current user turn
            prompt = _update_prompt(
                step.llm_prompt,
                original_user_steps[: i + 1],
                rephrased_user_steps[: i + 1],
            )
            if prompt:
                data.append(
                    _create_data_point(
                        prompt, step, conversation, rephrased_user_steps[i]
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

            # some user steps might have less rephrasings than others
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


def convert_to_fine_tuning_data(
    conversations: List[Conversation], storage_context: StorageContext
) -> List[LLMDataExample]:
    llm_data = []

    for i in tqdm(range(len(conversations))):
        llm_data.extend(_convert_conversation_into_llm_data(conversations[i]))

    storage_context.write_llm_data(
        llm_data, LLM_DATA_PREPARATION_MODULE_STORAGE_LOCATION
    )

    return llm_data
