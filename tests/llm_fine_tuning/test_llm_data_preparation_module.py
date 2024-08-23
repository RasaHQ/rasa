from typing import List
from unittest.mock import MagicMock

import pytest

from rasa.dialogue_understanding.commands import StartFlowCommand, SetSlotCommand
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.llm_data_preparation_module import (
    _construct_new_conversations,
    _update_prompt,
    _create_data_point,
    LLMDataExample,
    _convert_conversation_into_llm_data,
    convert_to_fine_tuning_data,
)


@pytest.fixture
def conversation() -> Conversation:
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

    return Conversation(
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
            ConversationStep(
                test_case.steps[2],
                [SetSlotCommand("amount", "50")],
                """
                Here is what happened previously in the conversation:
                USER: I want to send money to John
                AI: How much money do you want to send?
                USER: $50
                ===
                The user just said '''$50'''.
                """,
                [],
                ["It's 50 USD", "I owe him $50"],
            ),
            test_case.steps[3],
            ConversationStep(
                test_case.steps[4],
                [SetSlotCommand("confirmation", True)],
                """
                Here is what happened previously in the conversation:
                USER: I want to send money to John
                AI: How much money do you want to send?
                USER: $50
                AI: Do you want to send $50 to John?
                USER: yes
                ===
                The user just said '''yes'''.
                """,
                [],
                ["Yes, that is correct"],
            ),
        ],
        "transcript",
    )


@pytest.fixture
def conversation_mentioning_two_slots_upfront() -> Conversation:
    test_case = TestCase.from_dict(
        {
            "test_case": "transfer_money",
            "steps": [
                {"user": "I want to send 413$ to Maria"},
                {"bot": "Do you want to send 413$ to Maria?"},
                {"user": "yes"},
            ],
        }
    )

    return Conversation(
        test_case.name,
        test_case,
        [
            ConversationStep(
                test_case.steps[0],
                [
                    StartFlowCommand("transfer_money"),
                    SetSlotCommand("transfer_money_recipient", "Maria"),
                    SetSlotCommand("transfer_money_amount_of_money", "413"),
                ],
                """
                Here is what happened previously in the conversation:
                USER: I want to send 413$ to Maria
                ===
                The user just said '''I want to send 413$ to Maria'''.
                """,
                [],
                ["Send 413$ to Maria"],
            ),
            test_case.steps[1],
            ConversationStep(
                test_case.steps[2],
                [SetSlotCommand("confirmation", True)],
                """
                Here is what happened previously in the conversation:
                USER: I want to send 413$ to Maria
                AI: Do you want to send 413$ to Maria?
                USER: yes
                ===
                The user just said '''yes'''.
                """,
                [],
                ["Yes, that is correct"],
            ),
        ],
        "transcript",
    )


def test_construct_new_conversations(conversation: Conversation):
    new_conversations = _construct_new_conversations(conversation)

    assert len(new_conversations) == 2
    assert new_conversations[0] == [
        "Send money to John",
        "It's 50 USD",
        "Yes, that is correct",
    ]
    assert new_conversations[1] == [
        "Transfer money to John",
        "I owe him $50",
        "Yes, that is correct",
    ]


@pytest.mark.parametrize(
    "rephrasings, expected_conversations",
    (
        (
            [
                ["Send money to John", "Transfer money to John"],
                [],
                ["Yes, that is correct"],
            ],
            [
                [
                    "Send money to John",
                    "$50",
                    "Yes, that is correct",
                ],
                [
                    "Transfer money to John",
                    "$50",
                    "Yes, that is correct",
                ],
            ],
        ),
        ([[], [], []], []),
    ),
)
def test_construct_new_conversations_edge_cases(
    rephrasings: List[List[str]], expected_conversations: List[List[str]]
):
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
                "prompt",
                [],
                rephrasings[0],
            ),
            test_case.steps[1],
            ConversationStep(
                test_case.steps[2],
                [SetSlotCommand("amount", "50")],
                "prompt",
                [],
                rephrasings[1],
            ),
            test_case.steps[3],
            ConversationStep(
                test_case.steps[4],
                [SetSlotCommand("confirmation", True)],
                "prompt",
                [],
                rephrasings[2],
            ),
        ],
        "transcript",
    )

    new_conversations = _construct_new_conversations(conversation)

    assert len(new_conversations) == len(expected_conversations)
    for new_conversation, expected_converstaion in zip(
        new_conversations, expected_conversations
    ):
        assert new_conversation == expected_converstaion


def test_update_prompt(conversation: Conversation):
    prompt = """
    Here is what happened previously in the conversation:
    USER: I want to send money to John
    AI: How much money do you want to send?
    USER: $50
    AI: Do you want to send $50 to John?
    USER: yes
    ===
    The user just said '''yes'''.
    """
    original_user_steps = [
        step for step in conversation.iterate_over_annotated_user_steps()
    ]
    rephrased_user_steps = [
        "Transfer money to John",
        "I owe him $50",
        "Yes, that is correct",
    ]

    updated_prompt = _update_prompt(prompt, original_user_steps, rephrased_user_steps)

    assert (
        updated_prompt
        == """
    Here is what happened previously in the conversation:
    USER: Transfer money to John
    AI: How much money do you want to send?
    USER: I owe him $50
    AI: Do you want to send $50 to John?
    USER: Yes, that is correct
    ===
    The user just said '''Yes, that is correct'''.
    """
    )


def test_update_prompt_returns_none(conversation: Conversation):
    prompt = "prompt"
    original_user_steps = [
        step for step in conversation.iterate_over_annotated_user_steps()
    ]
    rephrased_user_steps = [
        "Transfer money to John",
        "Yes, that is correct",
    ]

    updated_prompt = _update_prompt(prompt, original_user_steps, rephrased_user_steps)

    assert updated_prompt is None


def test_create_data_point(conversation: Conversation):
    step = conversation.steps[0]
    prompt = """
    Here is what happened previously in the conversation:
    USER: I want to send money to John
    ===
    The user just said '''I want to send money to John'''.
    """
    rephrased_user_message = "Send money to John"

    data_point = _create_data_point(prompt, step, conversation, rephrased_user_message)

    assert isinstance(data_point, LLMDataExample)
    assert data_point.prompt == prompt
    assert data_point.output == step.commands_as_string()
    assert data_point.original_test_name == conversation.get_full_name()
    assert data_point.original_user_utterance == step.original_test_step.text
    assert data_point.rephrased_user_utterance == rephrased_user_message


def test_create_data_point_output_contains_multiple_commands(
    conversation_mentioning_two_slots_upfront: Conversation,
):
    step = conversation_mentioning_two_slots_upfront.steps[0]
    prompt = """
    Here is what happened previously in the conversation:
    USER: I want to send 413$ to Maria
    ===
    The user just said '''I want to send 413$ to Maria'''.
    """
    rephrased_user_message = "Send 413$ to Maria"

    data_point = _create_data_point(
        prompt, step, conversation_mentioning_two_slots_upfront, rephrased_user_message
    )

    assert isinstance(data_point, LLMDataExample)
    assert data_point.prompt == prompt
    assert (
        data_point.output
        == "StartFlow(transfer_money)\nSetSlot(transfer_money_recipient, Maria)\nSetSlot(transfer_money_amount_of_money, 413)"
    )  # noqa: E501
    assert (
        data_point.original_test_name
        == conversation_mentioning_two_slots_upfront.get_full_name()
    )
    assert data_point.original_user_utterance == step.original_test_step.text
    assert data_point.rephrased_user_utterance == rephrased_user_message


def test_convert_conversation_into_llm_data(conversation: Conversation):
    data = _convert_conversation_into_llm_data(conversation)

    assert len(data) == 9  # 3 original steps + 6 rephrased steps
    assert isinstance(data[0], LLMDataExample)
    assert data[0].prompt.strip().startswith("Here is what happened previously")
    assert data[0].original_user_utterance == "I want to send money to John"
    assert data[0].rephrased_user_utterance is None
    assert data[1].prompt.strip().startswith("Here is what happened previously")
    assert data[1].original_user_utterance == "I want to send money to John"
    assert data[1].rephrased_user_utterance == "Send money to John"
    assert data[2].prompt.strip().startswith("Here is what happened previously")
    assert data[2].original_user_utterance == "I want to send money to John"
    assert data[2].rephrased_user_utterance == "Transfer money to John"


def test_convert_to_fine_tuning_data(conversation: Conversation):
    storage_context = MagicMock()
    conversations = [conversation]

    llm_data = convert_to_fine_tuning_data(conversations, storage_context)

    assert len(llm_data) == 9  # 3 original steps + 6 rephrased steps
    assert storage_context.write_llm_data.called
