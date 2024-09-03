from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.paraphrasing.conversation_rephraser import (
    ConversationRephraser,
)
from rasa.llm_fine_tuning.paraphrasing_module import create_paraphrased_conversations
from rasa.shared.core.flows import FlowsList


@pytest.fixture
def mock_dependencies():
    mock_rephraser = Mock()
    mock_validator = Mock()
    mock_storage_context = Mock()

    mock_rephraser.rephrase_conversation = AsyncMock()
    mock_validator.validate_rephrasings = AsyncMock()

    return mock_rephraser, mock_validator, mock_storage_context


@pytest.fixture
def rephraser_config() -> Dict[str, Any]:
    return ConversationRephraser.get_default_config()


@pytest.fixture
def flows() -> FlowsList:
    return FlowsList([])


@pytest.fixture
def conversations() -> List[Conversation]:
    test_case = TestCase.from_dict(
        {
            "test_case": "transfer_money",
            "steps": [
                {"user": "I want to send money to John"},
                {"bot": "How much money do you want to send?"},
                {"user": "$50"},
                {"bot": "Do you want to send $50 to John?"},
                {"user": "yes"},
            ],
        }
    )

    conversation = Conversation(
        test_case.name,
        test_case,
        [
            ConversationStep(
                test_case.steps[0],
                [StartFlowCommand("transfer_money")],
                """
                Here is what happened previously in the conversation:
                USER: I want to send money to John
                ===
                The user just said '''I want to send money to John'''.
                """,
                [],
                ["Send money to John", "Transfer money to John"],
            ),
            test_case.steps[1],
        ],
        "transcript",
    )

    return [conversation]


@pytest.mark.asyncio
@patch(
    "rasa.llm_fine_tuning.paraphrasing.conversation_rephraser.ConversationRephraser",
    autospec=True,
)
@patch(
    "rasa.llm_fine_tuning.paraphrasing.rephrase_validator.RephraseValidator",
    autospec=True,
)
async def test_create_paraphrased_conversations_no_rephrases(
    MockRephraser,
    MockValidator,
    mock_dependencies,
    conversations,
    flows,
    rephraser_config,
):
    mock_rephraser, mock_validator, mock_storage_context = mock_dependencies
    MockRephraser.return_value = mock_rephraser
    MockValidator.return_value = mock_validator

    result_conversations, result_config = await create_paraphrased_conversations(
        conversations=conversations,
        rephrase_config=rephraser_config,
        num_rephrases=0,
        flows=flows,
        llm_command_generator_config={},
        storage_context=mock_storage_context,
    )

    assert result_conversations == conversations
    assert result_config == rephraser_config

    mock_rephraser.rephrase_conversation.assert_not_called()
    mock_validator.validate_rephrasings.assert_not_called()
    mock_storage_context.write_conversations.assert_not_called()
