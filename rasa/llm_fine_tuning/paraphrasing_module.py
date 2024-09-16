from typing import List, Dict, Any, Tuple

import structlog
from tqdm import tqdm

from rasa.llm_fine_tuning.conversations import Conversation
from rasa.llm_fine_tuning.paraphrasing.conversation_rephraser import (
    ConversationRephraser,
)
from rasa.llm_fine_tuning.paraphrasing.rephrase_validator import RephraseValidator
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.llm_fine_tuning.storage import StorageContext
from rasa.shared.core.flows import FlowsList
from rasa.shared.exceptions import ProviderClientAPIException

PARAPHRASING_MODULE_STORAGE_LOCATION = "2_rephrasings"

structlogger = structlog.get_logger()


async def create_paraphrased_conversations(
    conversations: List[Conversation],
    rephrase_config: Dict[str, Any],
    num_rephrases: int,
    flows: FlowsList,
    llm_command_generator_config: Dict[str, Any],
    storage_context: StorageContext,
) -> Tuple[List[Conversation], Dict[str, Any]]:
    """Create paraphrased conversations.

    Rephrase all user messages of a conversation and divide them into passing
    and failing rephrasings.

    Args:
        conversations: The conversations.
        rephrase_config: The path to the rephrase configuration file.
        num_rephrases: The number of rephrases to produce per user message.
        flows: All flows.
        llm_command_generator_config: The configuration of the trained model.
        storage_context: The storage context.

    Returns:
        The conversations including rephrasings and the configuration used for
        rephrasing.
    """
    rephraser = ConversationRephraser(rephrase_config)
    validator = RephraseValidator(llm_command_generator_config, flows)

    if num_rephrases <= 0:
        structlogger.info(
            "paraphrasing_module.skip",
            num_rephrases=num_rephrases,
            message="Skipping paraphrasing module as user messages should not be "
            "rephrased.",
        )
        return conversations, rephraser.config

    rephrased_conversations: List[Conversation] = []
    for i in tqdm(range(len(conversations))):
        current_conversation = conversations[i]

        try:
            # rephrase all user messages even if rephrase=False is set
            # to not confuse the LLM and get valid output
            rephrasings = await rephraser.rephrase_conversation(
                conversations[i], num_rephrases
            )
            # filter out the rephrasings for user messages that have rephrase=False set
            rephrasings = _filter_rephrasings(rephrasings, conversations[i])
            # check if the rephrasings are still producing the same commands
            rephrasings = await validator.validate_rephrasings(
                rephrasings, current_conversation
            )
        except ProviderClientAPIException as e:
            structlogger.error(
                "paraphrasing_module.skip_conversation",
                conversation=current_conversation.name,
                exception=str(e),
            )
            continue

        for j, step in enumerate(
            current_conversation.iterate_over_annotated_user_steps(rephrase=True)
        ):
            step.passed_rephrasings = rephrasings[j].passed_rephrasings
            step.failed_rephrasings = rephrasings[j].failed_rephrasings

        rephrased_conversations.append(current_conversation)

        storage_context.write_conversations(
            conversations, PARAPHRASING_MODULE_STORAGE_LOCATION
        )

    return rephrased_conversations, rephraser.config


def _filter_rephrasings(
    rephrasings: List[RephrasedUserMessage], conversation: Conversation
) -> List[RephrasedUserMessage]:
    """Filter rephrasings.

    Return only those rephrasings for user messages that have rephrase=True.

    Args:
        rephrasings: All rephrased user messages of the conversation.
        conversation: The conversation.

    Returns:
        Rephrasings for those user messages that have rephrase=True.
    """
    filtered_rephrasings = []
    index = 0
    user_messages = conversation.get_user_messages_to_rephrase()

    for rephrasing in rephrasings:
        if index >= len(user_messages):
            break

        # the user messages and the rephrasings are in the same order
        # rephrasings might contain more user messages as the user messages that
        # should be rephrased
        if rephrasing.original_user_message == user_messages[index]:
            filtered_rephrasings.append(rephrasing)
            index += 1

    return filtered_rephrasings
