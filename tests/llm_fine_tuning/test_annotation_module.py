from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, Mock

import pytest
from structlog.testing import capture_logs

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.e2e_test.e2e_test_case import TestSuite, TestCase, TestStep, ActualStepOutput
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.llm_fine_tuning.annotation_module import (
    annotate_e2e_tests,
    generate_conversation,
    _convert_to_conversation_step,
    _extract_llm_prompt_and_commands,
)
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.storage import StorageContext
from rasa.shared.core.events import UserUttered, BotUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import LLM_PROMPT, LLM_COMMANDS


@pytest.fixture
def test_step() -> TestStep:
    return TestStep.from_dict({"user": "I want to transfer money"})


@pytest.fixture
def test_turn(test_step: TestStep) -> ActualStepOutput:
    return ActualStepOutput.from_test_step(
        test_step,
        [
            UserUttered(
                "I want to transfer money",
                parse_data={
                    LLM_COMMANDS: [{"flow": "transfer_money", "command": "start flow"}],
                    LLM_PROMPT: "prompt",
                },
            ),
            BotUttered(
                "How much money do you want to transfer?",
                metadata={
                    "utter_action": "utter_ask_transfer_money_amount",
                },
            ),
        ],
    )


@patch("asyncio.run")
def test_annotate_e2e_tests(mock_asyncio_run: Mock):
    mock_runner = MagicMock(spec=E2ETestRunner)
    mock_test_suite = MagicMock(spec=TestSuite)
    mock_test_suite.test_cases = MagicMock(spec=List[TestCase])
    mock_test_suite.fixtures = []
    mock_test_suite.metadata = None
    mock_storage_context = MagicMock(spec=StorageContext)

    # Mock the return value of asyncio.run
    mock_conversations = [MagicMock(spec=Conversation), MagicMock(spec=Conversation)]
    mock_asyncio_run.return_value = mock_conversations

    # Call the function
    result = annotate_e2e_tests(mock_runner, mock_test_suite, mock_storage_context)

    # Assertions
    mock_runner.run_tests_for_fine_tuning.assert_called_once_with(
        mock_test_suite.test_cases, mock_test_suite.fixtures, mock_test_suite.metadata
    )

    mock_storage_context.write_conversations.assert_called_once_with(
        mock_conversations, "1_command_annotations"
    )

    assert result == mock_conversations


def test_generate_conversation(test_step: TestStep, test_turn: ActualStepOutput):
    test_turns = {0: test_step, 1: test_turn}
    test_case = TestCase("test_case_name", steps=[test_step, test_step])

    mock_tracker = MagicMock(spec=DialogueStateTracker)

    result = generate_conversation(test_turns, test_case, mock_tracker)

    assert result is not None
    assert isinstance(result, Conversation)
    assert result.original_e2e_test_case == test_case
    assert len(result.steps) == 2
    assert result.steps[0] == test_step
    assert isinstance(result.steps[1], ConversationStep)


def test_generate_conversation_using_assertions(
    test_step: TestStep, test_turn: ActualStepOutput
):
    test_turns = {0: test_turn, 1: test_turn}
    test_case = TestCase("test_case_name", steps=[test_step, test_step])

    mock_tracker = MagicMock(spec=DialogueStateTracker)

    result = generate_conversation(
        test_turns, test_case, mock_tracker, assertions_used=True
    )

    assert result is not None
    assert isinstance(result, Conversation)
    assert result.original_e2e_test_case == test_case
    assert len(result.steps) == 4
    assert isinstance(result.steps[0], ConversationStep)
    assert isinstance(result.steps[1], TestStep)
    assert isinstance(result.steps[2], ConversationStep)
    assert isinstance(result.steps[3], TestStep)


def test_convert_to_conversation_step_returns_conversation_step(
    test_step: TestStep, test_turn: ActualStepOutput
):
    test_case_name = "test_case"

    result = _convert_to_conversation_step(test_step, test_turn, test_case_name)

    assert isinstance(result, ConversationStep) is True
    assert result.llm_prompt == "prompt"
    assert result.llm_commands == [StartFlowCommand("transfer_money")]
    assert result.original_test_step == test_step


def test_convert_to_conversation_step_mismatch_between_test_step_and_test_turn(
    test_step: TestStep, test_turn: ActualStepOutput
):
    test_turn.text = "some other text"
    test_case_name = "test_case"

    with capture_logs() as logs:
        result = _convert_to_conversation_step(test_step, test_turn, test_case_name)

        assert len(logs) == 1
        assert logs[0]["log_level"] == "debug"
        assert logs[0]["test_case"] == test_case_name
        assert logs[0]["user_message"] == test_step.text

    assert isinstance(result, TestStep) is True
    assert result == test_step


def test_convert_to_conversation_step_no_command_prompt(
    test_step: TestStep,
):
    test_turn = ActualStepOutput.from_test_step(
        test_step, [UserUttered("I want to transfer money", parse_data={})]
    )
    test_case_name = "test_case"

    with capture_logs() as logs:
        result = _convert_to_conversation_step(test_step, test_turn, test_case_name)

        assert len(logs) == 1
        assert logs[0]["log_level"] == "debug"
        assert logs[0]["test_case"] == test_case_name
        assert logs[0]["user_message"] == test_step.text
        assert logs[0]["message"] == "No commands/prompt associated with the message."

    assert isinstance(result, TestStep) is True
    assert result == test_step


def test_extract_llm_prompt_and_commands(test_turn: ActualStepOutput):
    prompt, commands = _extract_llm_prompt_and_commands(test_turn)

    assert commands == [{"flow": "transfer_money", "command": "start flow"}]
    assert prompt == "prompt"


def test_extract_llm_prompt_and_commands_no_user_uttered_event(test_step: TestStep):
    test_turn = ActualStepOutput.from_test_step(test_step, [])

    prompt, commands = _extract_llm_prompt_and_commands(test_turn)

    assert commands is None
    assert prompt is None


@pytest.mark.parametrize(
    "parse_data",
    (
        {LLM_COMMANDS: [{"flow": "transfer_money", "command": "start flow"}]},
        {LLM_PROMPT: "prompt"},
        {},
    ),
)
def test_extract_llm_prompt_and_commands_no_commands_and_prompt(
    parse_data: Dict[str, Any], test_step: TestStep
):
    test_turn = ActualStepOutput.from_test_step(
        test_step, [UserUttered("I want to transfer money", parse_data=parse_data)]
    )

    prompt, commands = _extract_llm_prompt_and_commands(test_turn)

    assert commands is None
    assert prompt is None
