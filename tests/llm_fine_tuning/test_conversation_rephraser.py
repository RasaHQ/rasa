import asyncio
from typing import Any, Dict, Optional, List
from unittest.mock import patch, MagicMock, Mock

import pytest
from structlog.testing import capture_logs

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.paraphrasing.conversation_rephraser import (
    ConversationRephraser,
)
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.exceptions import ProviderClientAPIException

EXPECTED_PROMPT_PATH = "./tests/llm_fine_tuning/rendered_prompt.txt"


@pytest.fixture
def rephraser() -> ConversationRephraser:
    config = {
        "llm": {
            "type": "openai",
            "model_name": "gpt-3.5-turbo",
            "request_timeout": 7,
            "temperature": 0.0,
            "max_tokens": 4096,
        }
    }
    return ConversationRephraser(config)


@pytest.fixture
def conversation() -> Conversation:
    test_case = TestCase.from_dict(
        {
            "test_case": "transfer_money",
            "steps": [
                {"user": "I want to send money to John"},
                {"bot": "How much money do you want to send?"},
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
                "prompt",
            ),
            test_case.steps[1],
        ],
        "transcript",
    )


@patch(
    "rasa.llm_fine_tuning.paraphrasing.conversation_rephraser.ConversationRephraser._invoke_llm"
)
def test_rephrase_conversation(mock_invoke_llm: Mock, rephraser: ConversationRephraser):
    # Setup mock for invoke_llm
    mock_invoke_llm.return_value = "USER: How are you?\n0. How is everything?"

    # Create a mock Conversation object
    user_messages = ["How are you?"]
    mock_conversation = MagicMock(spec=Conversation)
    mock_conversation.get_user_messages.return_value = user_messages
    mock_conversation.name = "test_case"
    mock_conversation.transcript = "Test transcript"
    mock_conversation.iterate_over_annotated_user_steps.return_value = [
        MagicMock(original_test_step=MagicMock(text="How are you?"))
    ]

    # Call the method under test
    number_of_rephrasings = 1
    rephrased_messages = asyncio.run(
        rephraser.rephrase_conversation(mock_conversation, number_of_rephrasings)
    )

    # Assertions
    assert rephrased_messages is not None
    assert len(rephrased_messages) == len(user_messages)
    assert rephrased_messages[0].original_user_message == user_messages[0]
    assert len(rephrased_messages[0].rephrasings) == number_of_rephrasings
    assert rephrased_messages[0].rephrasings[0] == "How is everything?"


@patch("rasa.llm_fine_tuning.paraphrasing.conversation_rephraser.llm_factory")
def test_invoke_llm_failure(mock_llm_factory: Mock, rephraser: ConversationRephraser):
    # Mock the LLM to raise an exception
    mock_llm = MagicMock()
    mock_llm.apredict.side_effect = Exception("API error")
    mock_llm_factory.return_value = mock_llm

    # Call the private method _invoke_llm and assert it handles exception
    prompt = "test prompt"
    with pytest.raises(ProviderClientAPIException):
        result = asyncio.run(rephraser._invoke_llm(prompt))

        # Assertions
        assert result is None


@pytest.mark.parametrize(
    "block, expected_user_message, expected_rephrasings",
    (
        (
            """
            USER: Max
            1. My name is Max.
            2. I go by the name Max.
            3. I am identified as Max.
            """,
            "Max",
            ["My name is Max.", "I go by the name Max.", "I am identified as Max."],
        ),
        (
            """
            USER: Max
            1. USER: My name is Max.
            2. USER: I go by the name Max.
            3. USER: I am identified as Max.
            """,
            "Max",
            ["My name is Max.", "I go by the name Max.", "I am identified as Max."],
        ),
        (
            """
            Max
            1. My name is Max.
            2. I go by the name Max.
            3. I am identified as Max.
            """,
            "Max",
            ["My name is Max.", "I go by the name Max.", "I am identified as Max."],
        ),
        (
            """
            USER: Max
            - My name is Max.
            - I go by the name Max.
            - I am identified as Max.
            """,
            "Max",
            ["My name is Max.", "I go by the name Max.", "I am identified as Max."],
        ),
        (
            """

            """,
            None,
            None,
        ),
        (
            """
            USER: Max
            """,
            None,
            None,
        ),
    ),
)
def test_extract_rephrasings(
    block: str,
    expected_user_message: Optional[str],
    expected_rephrasings: Optional[List[str]],
    rephraser: ConversationRephraser,
):
    user_message, rephrasings = rephraser._extract_rephrasings(block)

    assert user_message == expected_user_message
    assert rephrasings == expected_rephrasings


@pytest.mark.parametrize(
    "output",
    (
        """
        USER: Show invoices
        1. I want to see my bills.
        2. I mean bills
        3. Yes, I want to see the invoices.

        USER: I'd like to book a car
        1. I need to reserve a car.
        2. Could I arrange for a car rental?
        3. I'm interested in hiring a car.
        """,
        """
        \"\"\"
        Show invoices
        1. I want to see my bills.
        2. I mean bills
        3. Yes, I want to see the invoices.

        USER: I'd like to book a car
        - I need to reserve a car.
        - Could I arrange for a car rental?
        - I'm interested in hiring a car.
        \"\"\"
        """,
        """USER: Show invoices
1. I want to see my bills.
2. I mean bills
3. Yes, I want to see the invoices.
USER: I'd like to book a car
1. I need to reserve a car.
2. Could I arrange for a car rental?
3. I'm interested in hiring a car.""",
    ),
)
def test_parse_output(output: str, rephraser: ConversationRephraser):
    number_of_rephrasings = 3
    user_messages = ["Show invoices", "I'd like to book a car"]

    rephrased_messages = rephraser._parse_output(output, user_messages)

    # Assertions
    assert rephrased_messages is not None
    assert len(rephrased_messages) == len(user_messages)
    assert rephrased_messages[0].original_user_message == user_messages[0]
    assert rephrased_messages[1].original_user_message == user_messages[1]
    assert len(rephrased_messages[0].rephrasings) == number_of_rephrasings
    assert rephrased_messages[0].rephrasings[0] == "I want to see my bills."


def test_parse_output_insufficient_rephrasings(rephraser: ConversationRephraser):
    output = """
    USER: Show invoices
    1. I want to see my bills.
    2. I mean bills
    3. Yes, I want to see the invoices.

    USER: I'd like to book a car
    1. I need to reserve a car.
    3. I'm interested in hiring a car.
    """
    user_messages = ["Show invoices", "I'd like to book a car"]

    rephrased_messages = rephraser._parse_output(output, user_messages)

    # Assertions
    assert rephrased_messages is not None
    assert len(rephrased_messages) == len(user_messages)
    assert len(rephrased_messages[0].rephrasings) == 3
    assert len(rephrased_messages[1].rephrasings) == 2


@pytest.mark.parametrize(
    "input_message",
    (
        "USER: invalid user message",
        "USER Show invoices",
        "USER - Show invoices",
    ),
)
def test_parse_output_invalid_user_message(input_message: str):
    output = f"""
    {input_message}
    1. I want to see my bills.

    USER: I'd like to book a car
    1. I need to reserve a car.
    """
    user_messages = ["Show invoices", "I'd like to book a car"]

    rephraser = ConversationRephraser({})
    rephrased_messages = rephraser._parse_output(output, user_messages)

    expected_rephrase_message = [
        RephrasedUserMessage(
            original_user_message="Show invoices",
            rephrasings=[],
        ),
        RephrasedUserMessage(
            original_user_message="I'd like to book a car",
            rephrasings=["I need to reserve a car."],
        ),
    ]

    assert rephrased_messages == expected_rephrase_message


def test_render_template(rephraser: ConversationRephraser, conversation: Conversation):
    with open(EXPECTED_PROMPT_PATH, "r", encoding="unicode_escape") as f:
        expected_prompt = f.readlines()
        expected_prompt = "".join(expected_prompt)

    prompt = rephraser._render_template(conversation, 10)

    assert prompt == expected_prompt


def test_check_rephrasings_with_invalid_number_of_rephrasings(
    rephraser: ConversationRephraser,
):
    conversation_name = "name"
    llm_output = "llm output"
    number_of_rephrasings = 3
    rephrased_messages = [
        RephrasedUserMessage("message 1", ["rephrase_2"]),
        RephrasedUserMessage("message 2", ["rephrase_1", "rephrase_2", "rephrase_3"]),
    ]

    with capture_logs() as logs:
        rephraser._check_rephrasings(
            rephrased_messages, number_of_rephrasings, llm_output, conversation_name
        )

        assert len(logs) == 1
        assert logs[0]["conversation_name"] == conversation_name
        assert logs[0]["llm_output"] == llm_output
        assert logs[0]["log_level"] == "warning"
        assert (
            logs[0]["event"]
            == "conversation_rephraser.rephrase_conversation.parse_llm_output"
        )
        assert logs[0]["incorrect_rephrasings_for_messages"] == ["message 1"]


def test_check_rephrasings_all_good(rephraser: ConversationRephraser):
    conversation_name = "name"
    llm_output = "llm output"
    number_of_rephrasings = 2
    rephrased_messages = [
        RephrasedUserMessage("message 1", ["rephrase_1", "rephrase_2"]),
        RephrasedUserMessage("message 2", ["rephrase_1", "rephrase_2"]),
    ]

    with capture_logs() as logs:
        rephraser._check_rephrasings(
            rephrased_messages, number_of_rephrasings, llm_output, conversation_name
        )

        assert len(logs) == 0


@pytest.mark.parametrize(
    "output",
    (
        """
        USER: Show invoices
        1. I want to see my bills.
        2. I mean bills
        3. Yes, I want to see the invoices.

        USER: I'd like to book a car
        1. I need to reserve a car.
        2. Could I arrange for a car rental?
        3. I'm interested in hiring a car.
        """,
        """
        USER: Show invoices
        1. I want to see my bills.
        2. I mean bills
        3. Yes, I want to see the invoices.


        USER: I'd like to book a car
        1. I need to reserve a car.
        2. Could I arrange for a car rental?
        3. I'm interested in hiring a car.
        """,
        """
        \"\"\"
        Show invoices
        1. I want to see my bills.
        2. I mean bills
        3. Yes, I want to see the invoices.

        USER: I'd like to book a car
        - I need to reserve a car.
        - Could I arrange for a car rental?
        - I'm interested in hiring a car.
        \"\"\"
        """,
        """USER: Show invoices
1. I want to see my bills.
2. I mean bills
3. Yes, I want to see the invoices.
USER: I'd like to book a car
1. I need to reserve a car.
2. Could I arrange for a car rental?
3. I'm interested in hiring a car.""",
    ),
)
def test_get_message_blocks(output: str, rephraser: ConversationRephraser):
    message_blocks = rephraser._get_message_blocks(output, 2)

    # Assertions
    assert len(message_blocks) == 2


def test_get_message_blocks_returns_empty_list(rephraser: ConversationRephraser):
    output = """USER: Show invoices
            1. I want to see my bills.
            2. I mean bills
            3. Yes, I want to see the invoices.
            &&&
            USER: I'd like to book a car
            1. I need to reserve a car.
            2. Could I arrange for a car rental?
            3. I'm interested in hiring a car.
            """

    message_blocks = rephraser._get_message_blocks(output, 2)

    # Assertions
    assert not message_blocks


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"llm": {"type": "some_type", "model_name": "gpt-xyz"}},
        {"llm": {"type": "some_type", "model": "gpt-xyz"}},
        {"llm": {"model": "gpt-xyz"}, "prompt_template": "test"},
        {"prompt_template": "test"},
    ],
)
def test_validate_config_passes_validation(config: Dict[str, Any]) -> None:
    ConversationRephraser.validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"llm": {}},
        {"llm": {"type": "some_type"}},
        {"llm": {"model_name": "gpt-xyz"}, "some_key": ""},
        {"llm": {"model_name": "gpt-xyz"}, "some_key": "", "prompt_template": "test"},
        {"some_key": "", "prompt_template": "test"},
    ],
)
def test_validate_config_fails_validation(config: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ConversationRephraser.validate_config(config)
