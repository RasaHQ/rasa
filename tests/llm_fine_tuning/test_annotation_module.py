from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from structlog.testing import capture_logs

from rasa.dialogue_understanding.commands import StartFlowCommand
from rasa.e2e_test.e2e_test_case import ActualStepOutput, TestCase, TestStep, TestSuite
from rasa.e2e_test.e2e_test_runner import E2ETestRunner
from rasa.llm_fine_tuning.annotation_module import (
    _convert_to_conversation_step,
    _extract_llm_prompt_and_commands,
    _should_be_rephrased,
    annotate_e2e_tests,
    generate_conversation,
)
from rasa.llm_fine_tuning.conversations import Conversation, ConversationStep
from rasa.llm_fine_tuning.storage import StorageContext
from rasa.shared.core.events import BotUttered, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import LLM_COMMANDS, LLM_PROMPT


@pytest.fixture
def test_step() -> TestStep:
    return TestStep.from_dict({"user": "I want to transfer money"})


@pytest.fixture
def test_step_bot() -> TestStep:
    return TestStep.from_dict({"bot": "How much money do you want to transfer?"})


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


@pytest.fixture
def test_turn_from_buttons(test_step: TestStep) -> ActualStepOutput:
    return ActualStepOutput.from_test_step(
        TestStep.from_dict({"user": "to John"}),
        [
            BotUttered(
                "Do you want to transfer that money?",
                data={
                    "buttons": [
                        {"title": "yes", "payload": "yes"},
                        {"title": "no", "payload": "no"},
                    ]
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


@pytest.mark.parametrize(
    "events,steps,test_turns,expected_len,assertions_used,expected_step_types",
    [
        # Only user utterances; no assertions
        (
            [UserUttered("I want to transfer money")] * 2,
            lambda test_step, test_step_bot: [test_step, test_step],
            lambda test_step, test_step_bot, test_turn: {0: test_step, 1: test_turn},
            2,
            False,
            [TestStep, ConversationStep],
        ),
        # User and bot utterance; no assertions
        (
            [
                UserUttered("I want to transfer money"),
                BotUttered("How much money do you want to transfer?"),
            ],
            lambda test_step, test_step_bot: [test_step, test_step_bot],
            lambda test_step, test_step_bot, test_turn: {
                0: test_turn,
                1: test_step_bot,
            },
            2,
            False,
            [ConversationStep, TestStep],
        ),
        # User, bot, user utterances; no assertions
        (
            [
                UserUttered("I want to transfer money"),
                BotUttered("How much money do you want to transfer?"),
                UserUttered("I want to transfer money"),
            ],
            lambda test_step, test_step_bot: [test_step, test_step_bot, test_step],
            lambda test_step, test_step_bot, test_turn: {
                0: test_turn,
                1: test_step_bot,
            },
            2,
            False,
            [ConversationStep, TestStep],
        ),
        # User, bot, user, user utterances; no assertions
        (
            [
                UserUttered("I want to transfer money"),
                BotUttered("How much money do you want to transfer?"),
                UserUttered("I want to transfer money"),
                UserUttered("I want to transfer money"),
            ],
            lambda test_step, test_step_bot: [
                test_step,
                test_step_bot,
                test_step,
                test_step,
            ],
            lambda test_step, test_step_bot, test_turn: {
                0: test_turn,
                1: test_step_bot,
            },
            2,
            False,
            [ConversationStep, TestStep],
        ),
        # Only user utterances; assertions used
        (
            [UserUttered("I want to transfer money")] * 2,
            lambda test_step, test_step_bot: [test_step, test_step],
            lambda test_step, test_step_bot, test_turn: {0: test_turn, 1: test_turn},
            4,
            True,
            [ConversationStep, TestStep],
        ),
        # Multiple, different user utterances in test case; assertions used
        # With assertions, the test case steps consist of user utterances only,
        # whilst their corresponding, expected bot utterances are defined as assertions.
        (
            [
                UserUttered("I want to transfer money"),
                # followed by bot utterance in the form of an assertion
                # bot_uttered: "Who do you want to transfer money to?",
                UserUttered("to John"),
                # followed by bot utterance in the form of an assertion
                # bot_uttered: "Do you want to transfer that money?",
                UserUttered("yes"),
            ],
            lambda test_step, test_step_bot: [test_step, test_step, test_step],
            lambda test_step, test_step_bot, test_turn: {
                0: test_turn,
                1: test_turn,
                2: test_turn,
            },
            6,  # 3 * 2, since each user step is paired with an assertion step
            True,
            # alternates between ConversationStep and TestStep for each pair
            [
                ConversationStep,
                TestStep,
                ConversationStep,
                TestStep,
                ConversationStep,
                TestStep,
            ],
        ),
    ],
)
def test_generate_conversation_various_cases(
    test_step: TestStep,
    test_step_bot: TestStep,
    test_turn: ActualStepOutput,
    events,
    steps,
    test_turns,
    expected_len,
    assertions_used,
    expected_step_types,
):
    tracker = DialogueStateTracker.from_events("test_annotation_module", events)
    test_case = TestCase("test_case_name", steps=steps(test_step, test_step_bot))
    turns = test_turns(test_step, test_step_bot, test_turn)

    result = generate_conversation(
        turns, test_case, tracker, assertions_used=assertions_used
    )

    assert result is not None
    assert isinstance(result, Conversation)
    assert result.original_e2e_test_case == test_case
    assert len(result.steps) == expected_len
    for idx, expected_step_type in enumerate(expected_step_types):
        assert isinstance(result.steps[idx], expected_step_type)


def test_convert_to_conversation_step_returns_conversation_step(
    test_step: TestStep, test_turn: ActualStepOutput
):
    test_case_name = "test_case"

    result = _convert_to_conversation_step(test_step, test_turn, test_case_name, None)

    assert isinstance(result, ConversationStep) is True
    assert result.llm_prompt == "prompt"
    assert result.llm_commands == [StartFlowCommand("transfer_money")]
    assert result.original_test_step == test_step
    assert result.rephrase is True


@pytest.mark.parametrize(
    "user_message, rephrase",
    (
        ("some other user message", True),
        ("yes", False),
        ("Yes", False),
        ("no", False),
        ("NO", False),
    ),
)
def test_convert_to_conversation_step_returns_conversation_step_with_rephrase_false(
    test_turn: ActualStepOutput,
    test_turn_from_buttons: ActualStepOutput,
    user_message: str,
    rephrase: bool,
):
    test_case_name = "test_case"
    test_turn.text = user_message
    test_step = TestStep.from_dict({"user": user_message})

    result = _convert_to_conversation_step(
        test_step, test_turn, test_case_name, test_turn_from_buttons
    )

    assert isinstance(result, ConversationStep) is True
    assert result.llm_prompt == "prompt"
    assert result.llm_commands == [StartFlowCommand(flow="transfer_money")]
    assert result.original_test_step == test_step
    assert result.rephrase == rephrase


def test_convert_to_conversation_step_mismatch_between_test_step_and_test_turn(
    test_step: TestStep, test_turn: ActualStepOutput
):
    test_turn.text = "some other text"
    test_case_name = "test_case"

    with capture_logs() as logs:
        result = _convert_to_conversation_step(
            test_step, test_turn, test_case_name, None
        )

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
        result = _convert_to_conversation_step(
            test_step, test_turn, test_case_name, None
        )

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


@pytest.mark.parametrize(
    "user_message, previous_turn, expected_result",
    (
        (
            "yes",
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "to John"}),
                [
                    BotUttered(
                        "Do you want to transfer that money?",
                        data={
                            "buttons": [
                                {"title": "yes", "payload": "yes"},
                                {"title": "no", "payload": "no"},
                            ]
                        },
                    ),
                ],
            ),
            False,
        ),
        (
            "YES",
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "to John"}),
                [
                    BotUttered(
                        "Do you want to transfer that money?",
                        data={
                            "buttons": [
                                {"title": "yes", "payload": "yes"},
                                {"title": "no", "payload": "no"},
                            ]
                        },
                    ),
                ],
            ),
            False,
        ),
        (
            "some other text",
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "to John"}),
                [
                    BotUttered(
                        "Do you want to transfer that money?",
                        data={
                            "buttons": [
                                {"title": "yes", "payload": "yes"},
                                {"title": "no", "payload": "no"},
                            ]
                        },
                    ),
                ],
            ),
            True,
        ),
        ("yes", None, True),
        (
            "yes",
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "to John"}),
                [],
            ),
            True,
        ),
        (
            "yes",
            ActualStepOutput.from_test_step(
                TestStep.from_dict({"user": "to John"}),
                [
                    BotUttered(
                        "Do you want to transfer that money?",
                    )
                ],
            ),
            True,
        ),
    ),
)
def test_should_be_rephrased(
    user_message: str, previous_turn: Optional[ActualStepOutput], expected_result: bool
):
    rephrase = _should_be_rephrased(user_message, previous_turn, "test_case_name")

    assert rephrase == expected_result
