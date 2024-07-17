import asyncio
from typing import List
from unittest.mock import patch, Mock, MagicMock

import pytest

from rasa.dialogue_understanding.commands import (
    SetSlotCommand,
    StartFlowCommand,
    ClarifyCommand,
    CancelFlowCommand,
    SkipQuestionCommand,
    HumanHandoffCommand,
    ChitChatAnswerCommand,
    KnowledgeAnswerCommand,
    Command,
)
from rasa.e2e_test.e2e_test_case import TestCase
from rasa.llm_fine_tuning.conversations import ConversationStep, Conversation
from rasa.llm_fine_tuning.paraphrasing.rephrase_validator import RephraseValidator
from rasa.llm_fine_tuning.paraphrasing.rephrased_user_message import (
    RephrasedUserMessage,
)
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.exceptions import ProviderClientAPIException


@pytest.fixture
def validator() -> RephraseValidator:
    config = {
        "type": "openai",
        "model_name": "gpt-3.5-turbo",
        "request_timeout": 7,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    flows = flows_from_str(
        """
        flows:
          transfer_money:
            description: send money to a recipient
            steps:
            - collect: recipient
            - collect: amount
          book_hotel:
            description: book a hotel
            steps:
            - collect: hotel_name
            - collect: start_date
            - collect: end_date
        """
    )
    return RephraseValidator(config, flows)


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
                """
                Here is what happened previously in the conversation:
                USER: I want to send money to John
                AI: How much money do you want to send?
                ===
                The user just said '''I want to send money to John'''.
                """,
            ),
            test_case.steps[1],
        ],
        "transcript",
    )


@pytest.fixture
def rephrased_user_messages() -> List[RephrasedUserMessage]:
    return [
        RephrasedUserMessage(
            "I want to send money to John",
            ["Send money to John", "Transfer money to John"],
        )
    ]


@patch("rasa.llm_fine_tuning.paraphrasing.rephrase_validator.llm_factory")
def test_invoke_llm_failure(mock_llm_factory: Mock, validator: RephraseValidator):
    # Mock the LLM to raise an exception
    mock_llm = MagicMock()
    mock_llm.apredict.side_effect = Exception("API error")
    mock_llm_factory.return_value = mock_llm

    # Call the private method _invoke_llm and assert it handles exception
    prompt = "test prompt"
    with pytest.raises(ProviderClientAPIException):
        result = asyncio.run(validator._invoke_llm(prompt))

        # Assertions
        assert result is None


def test_validate_rephrasings_passing(
    validator: RephraseValidator,
    conversation: Conversation,
    rephrased_user_messages: List[RephrasedUserMessage],
):
    with patch.object(validator, "_validate_rephrase_is_passing", return_value=True):
        validated_rephrasings = asyncio.run(
            validator.validate_rephrasings(rephrased_user_messages, conversation)
        )

        assert len(validated_rephrasings[0].passed_rephrasings) == 2
        assert "Send money to John" in validated_rephrasings[0].passed_rephrasings
        assert "Transfer money to John" in validated_rephrasings[0].passed_rephrasings
        assert len(validated_rephrasings[0].failed_rephrasings) == 0


def test_validate_rephrasings_failing(
    validator: RephraseValidator,
    conversation: Conversation,
    rephrased_user_messages: List[RephrasedUserMessage],
):
    with patch.object(validator, "_validate_rephrase_is_passing", return_value=False):
        validated_rephrasings = asyncio.run(
            validator.validate_rephrasings(rephrased_user_messages, conversation)
        )

        assert len(validated_rephrasings[0].failed_rephrasings) == 2
        assert "Send money to John" in validated_rephrasings[0].failed_rephrasings
        assert "Transfer money to John" in validated_rephrasings[0].failed_rephrasings
        assert len(validated_rephrasings[0].passed_rephrasings) == 0


@patch(
    "rasa.llm_fine_tuning.paraphrasing.rephrase_validator.RephraseValidator."
    "_invoke_llm"
)
@patch(
    "rasa.dialogue_understanding.generator.single_step."
    "single_step_llm_command_generator.SingleStepLLMCommandGenerator.parse_commands"
)
def test_rephrase_is_passing(
    mock_parse_commands: Mock,
    mock_invoke_llm: Mock,
    validator: RephraseValidator,
    conversation: Conversation,
):
    mock_invoke_llm.return_value = "StartFlow(transfer_money)"
    mock_parse_commands.return_value = [StartFlowCommand("transfer_money")]

    rephrase = "I want to transfer some money to John"
    passing = asyncio.run(
        validator._validate_rephrase_is_passing(rephrase, conversation.steps[0])
    )

    assert passing is True


@patch(
    "rasa.llm_fine_tuning.paraphrasing.rephrase_validator.RephraseValidator."
    "_invoke_llm"
)
@patch(
    "rasa.dialogue_understanding.generator.single_step."
    "single_step_llm_command_generator.SingleStepLLMCommandGenerator.parse_commands"
)
def test_rephrase_is_not_passing(
    mock_parse_commands: Mock,
    mock_invoke_llm: Mock,
    validator: RephraseValidator,
    conversation: Conversation,
):
    mock_invoke_llm.return_value = "SetSlot('recipient', 'John')"
    mock_parse_commands.return_value = [SetSlotCommand("recipient", "John")]

    rephrase = "I want to transfer some money to John"
    passing = asyncio.run(
        validator._validate_rephrase_is_passing(rephrase, conversation.steps[0])
    )

    assert passing is False


@pytest.mark.parametrize(
    "expected_commands, actual_commands, match",
    [
        ([StartFlowCommand("foo")], [StartFlowCommand("foo")], True),
        (
            [ClarifyCommand(options=["a", "b", "c"])],
            [ClarifyCommand(options=["a", "b", "c"])],
            True,
        ),
        ([CancelFlowCommand()], [CancelFlowCommand()], True),
        ([SkipQuestionCommand()], [SkipQuestionCommand()], True),
        ([HumanHandoffCommand()], [HumanHandoffCommand()], True),
        ([ChitChatAnswerCommand()], [ChitChatAnswerCommand()], True),
        ([KnowledgeAnswerCommand()], [KnowledgeAnswerCommand()], True),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("foo", "bar")], True),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("bar", "foo")], False),
        ([SetSlotCommand("foo", "bar")], [SetSlotCommand("foo", "BAR")], True),
        (
            [ChitChatAnswerCommand(), StartFlowCommand("foo")],
            [ChitChatAnswerCommand()],
            False,
        ),
        (
            [KnowledgeAnswerCommand()],
            [KnowledgeAnswerCommand(), StartFlowCommand("foo")],
            False,
        ),
    ],
)
def test_commands_match(
    validator: RephraseValidator,
    expected_commands: List[Command],
    actual_commands: List[Command],
    match: bool,
):
    assert validator._check_commands_match(expected_commands, actual_commands) is match


def test_update_prompt(validator: RephraseValidator):
    original = "I want to send money to John"
    rephrased = "SOME REPHRASE TEXT"
    prompt = """
        Here is what happened previously in the conversation:
        USER: I want to send money to John
        AI: How much money do you want to send?
        ===
        The user just said '''I want to send money to John'''.
        """

    updated_prompt = validator._update_prompt(rephrased, original, prompt)
    assert (
        updated_prompt
        == """
        Here is what happened previously in the conversation:
        USER: SOME REPHRASE TEXT
        AI: How much money do you want to send?
        ===
        The user just said '''SOME REPHRASE TEXT'''.
        """
    )
