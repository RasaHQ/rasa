from typing import Optional, List, Text, Type
from unittest.mock import Mock, patch

import pytest
from rasa.dialogue_understanding.commands import Command, StartFlowCommand, ErrorCommand
from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.dialogue_understanding.generator.command_generator import CommandGenerator
from rasa.shared.constants import (
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
    RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY,
)
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT, COMMANDS
from rasa.shared.nlu.training_data.message import Message


class WackyCommandGenerator(CommandGenerator):
    async def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        if message.get(TEXT) == "Hi":
            raise ValueError("Message too banal - I am quitting.")
        else:
            return [ChitChatAnswerCommand()]


async def test_command_generator_catches_processing_errors():
    generator = WackyCommandGenerator({})
    messages = [Message.build("Hi"), Message.build("What is your purpose?")]
    await generator.process(messages, FlowsList(underlying_flows=[]))
    commands = [m.get(COMMANDS) for m in messages]

    assert len(commands[0]) == 0
    assert len(commands[1]) == 1
    assert commands[1][0]["command"] == ChitChatAnswerCommand.command()


async def test_command_generator_still_throws_not_implemented_error():
    # This test can be removed if the predict_commands method stops to be abstract
    generator = CommandGenerator({})
    with pytest.raises(NotImplementedError):
        await generator.process([Message.build("test")], FlowsList([]))


@pytest.mark.parametrize(
    "commands, startable_flows, expected_commands",
    [
        ([ChitChatAnswerCommand()], FlowsList([]), [ChitChatAnswerCommand()]),
        (
            [StartFlowCommand("spam")],
            FlowsList([Flow("spam")]),
            [StartFlowCommand("spam")],
        ),
        ([StartFlowCommand("eggs")], FlowsList([Flow("spam")]), []),
        (
            [StartFlowCommand("spam"), StartFlowCommand("eggs")],
            FlowsList([Flow("spam")]),
            [StartFlowCommand("spam")],
        ),
    ],
)
def test_check_commands_against_startable_flows(
    commands: List[Command],
    startable_flows: FlowsList,
    expected_commands: List[Command],
):
    """Test that commands are correctly filtered against startable flows."""
    # Given
    generator = CommandGenerator({})
    # When
    commands = generator._check_commands_against_startable_flows(
        commands, startable_flows
    )
    # Then
    assert commands == expected_commands


@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._check_commands_against_startable_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_startable_flows"
)
async def test_command_processor_checks_flow_guards(
    mock_get_startable_flows: Mock, mock_check_commands_against_startable_flows: Mock
):
    """Test that the command processor checks flow guards."""
    # Given
    generator = WackyCommandGenerator({})
    messages = [Message.build("What is your purpose?")]
    # When
    await generator.process(messages, FlowsList([Flow("spam")]))
    # Then
    mock_get_startable_flows.assert_called_once()
    mock_check_commands_against_startable_flows.assert_called_once()


@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._check_commands_against_startable_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_startable_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_active_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator._evaluate_and_predict"
)
async def test_command_generator_adds_active_flows_to_avaialble_flows(
    mock_evaluate_and_predict: Mock,
    mock_get_active_flows: Mock,
    mock_get_startable_flows: Mock,
    mock_check_commands_against_startable_flows: Mock,
):
    """Test that the command processor checks flow guards."""
    # Given
    generator = WackyCommandGenerator({})
    messages = [Message.build("What is your purpose?")]

    mock_tracker = Mock()

    mock_get_active_flows.return_value = FlowsList(
        [Flow("active_flow"), Flow("called_flow")]
    )
    mock_get_startable_flows.return_value = FlowsList(
        [Flow("startable_flow_1"), Flow("startable_flow_2")]
    )

    # When
    await generator.process(
        messages,
        FlowsList(
            [
                Flow("other_flow_1"),
                Flow("other_flow_2"),
                Flow("active_flow"),
                Flow("called_flow"),
                Flow("startable_flow_1"),
                Flow("startable_flow_2"),
            ]
        ),
        mock_tracker,
    )
    # Then
    mock_evaluate_and_predict.assert_called_once_with(
        messages[0],
        FlowsList(
            [
                Flow("startable_flow_1"),
                Flow("startable_flow_2"),
                Flow("active_flow"),
                Flow("called_flow"),
            ]
        ),
        mock_tracker,
    )
    mock_get_startable_flows.assert_called_once()
    mock_get_active_flows.assert_called_once()
    mock_check_commands_against_startable_flows.assert_called_once()


@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_active_flows"
)
@patch(
    "rasa.dialogue_understanding.generator.command_generator"
    ".CommandGenerator.get_startable_flows"
)
async def test_process_does_not_predict_commands_if_commands_already_present(
    mock_get_active_flows: Mock,
    mock_startable_flows: Mock,
):
    """Test that predict_commands does not overwrite commands
    if commands are already set on message."""
    command_generator = CommandGenerator({})

    command = StartFlowCommand("some flow").as_dict()

    test_message = Message.build(text="some message")
    test_message.set(COMMANDS, [command], add_to_output=True)

    assert len(test_message.get(COMMANDS)) == 1
    assert test_message.get(COMMANDS) == [command]

    mock_tracker = Mock()
    mock_tracker.get_slot = Mock(return_value=[])

    mock_get_active_flows.return_value = FlowsList([])
    mock_startable_flows.return_value = FlowsList([Flow("startable_flow")])

    returned_message = (
        await command_generator.process(
            [test_message],
            flows=FlowsList([Flow("startable_flow")]),
            tracker=mock_tracker,
        )
    )[0]

    assert len(returned_message.get(COMMANDS)) == 1
    assert returned_message.get(COMMANDS) == [command]


@pytest.mark.parametrize(
    "message, expected_error_type",
    [
        (" \n\t", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("Very long message", RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG),
    ],
)
async def test_process_invalid_messages(message: Text, expected_error_type: Text):
    # Given
    generator = WackyCommandGenerator({"user_input": {"max_characters": 10}})
    # When
    processed_messages = await generator.process(
        messages=[Message.build(text=message)], flows=FlowsList(underlying_flows=[])
    )
    # Then
    command = processed_messages[0].data["commands"][0]
    assert command["command"] == ErrorCommand.command()
    assert command["error_type"] == expected_error_type


@pytest.mark.parametrize(
    "message, expected_command_type, expected_error_type",
    [
        (" \n\t", ErrorCommand, RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        ("", ErrorCommand, RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_EMPTY),
        (
            "Very very long message",
            ErrorCommand,
            RASA_PATTERN_INTERNAL_ERROR_USER_INPUT_TOO_LONG,
        ),
        ("Chit Chat", ChitChatAnswerCommand, None),
    ],
)
async def test_evaluate_and_predict_commands(
    message: Text, expected_command_type: Type, expected_error_type: Optional[Text]
):
    # Given
    generator = WackyCommandGenerator({"user_input": {"max_characters": 20}})
    # When
    commands = await generator._evaluate_and_predict(
        message=Message.build(text=message),
        startable_flows=FlowsList(underlying_flows=[]),
    )
    # Then
    assert len(commands) == 1
    assert isinstance(commands[0], expected_command_type)
    if isinstance(commands[0], ErrorCommand):
        assert commands[0].error_type == expected_error_type
