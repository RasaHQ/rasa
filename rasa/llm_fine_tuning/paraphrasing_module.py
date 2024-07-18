from typing import List, Dict, Any, Tuple, Optional

import structlog
from rasa.llm_fine_tuning.conversation_storage import StorageContext
from rasa.llm_fine_tuning.conversations import Conversation
from rasa.llm_fine_tuning.paraphrasing.conversation_rephraser import (
    ConversationRephraser,
)
from rasa.llm_fine_tuning.paraphrasing.rephrase_validator import RephraseValidator
from rasa.shared.core.flows import FlowsList
from rasa.shared.exceptions import ProviderClientAPIException
from rasa.shared.utils.yaml import read_config_file
from tqdm import tqdm

PARAPHRASING_MODULE_STORAGE_LOCATION = "2_rephrasings"

structlogger = structlog.get_logger()


async def create_paraphrased_conversations(
    conversations: List[Conversation],
    rephrase_config_path: Optional[str],
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
        rephrase_config_path: The path to the rephrase configuration file.
        num_rephrases: The number of rephrases to produce per user message.
        flows: All flows.
        llm_command_generator_config: The configuration of the trained model.
        storage_context: The storage context.

    Returns:
        The conversations including rephrasings and the configuration used for
        rephrasing.
    """
    config = read_config_file(rephrase_config_path) if rephrase_config_path else {}

    rephraser = ConversationRephraser(config)
    validator = RephraseValidator(llm_command_generator_config, flows)

    rephrased_conversations: List[Conversation] = []
    for i in tqdm(range(len(conversations))):
        current_conversation = conversations[i]

        try:
            rephrasings = await rephraser.rephrase_conversation(
                conversations[i], num_rephrases
            )
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
            current_conversation.iterate_over_annotated_user_steps()
        ):
            step.passed_rephrasings = rephrasings[j].passed_rephrasings
            step.failed_rephrasings = rephrasings[j].failed_rephrasings

        rephrased_conversations.append(current_conversation)

        storage_context.write_conversations(
            conversations, PARAPHRASING_MODULE_STORAGE_LOCATION
        )

    return rephrased_conversations, rephraser.config
