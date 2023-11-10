from typing import Optional, List
from unittest.mock import Mock, patch

import pytest

from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.dialogue_understanding.generator.command_generator import CommandGenerator
from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows import Flow, FlowsList
from rasa.shared.core.slots import AnySlot, BooleanSlot, TextSlot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import TEXT, COMMANDS
from rasa.shared.nlu.training_data.message import Message


class WackyCommandGenerator(CommandGenerator):
    def predict_commands(
        self,
        message: Message,
        flows: FlowsList,
        tracker: Optional[DialogueStateTracker] = None,
    ) -> List[Command]:
        if message.get(TEXT) == "Hi":
            raise ValueError("Message too banal - I am quitting.")
        else:
            return [ChitChatAnswerCommand()]


def test_command_generator_catches_processing_errors():
    generator = WackyCommandGenerator()
    messages = [Message.build("Hi"), Message.build("What is your purpose?")]
    generator.process(messages, FlowsList(underlying_flows=[]))
    commands = [m.get(COMMANDS) for m in messages]

    assert len(commands[0]) == 0
    assert len(commands[1]) == 1
    assert commands[1][0]["command"] == ChitChatAnswerCommand.command()


def test_command_generator_still_throws_not_implemented_error():
    # This test can be removed if the predict_commands method stops to be abstract
    generator = CommandGenerator()
    with pytest.raises(NotImplementedError):
        generator.process([Message.build("test")], FlowsList([]))


def test_get_context_and_slots():
    """Test that context and slots are correctly extracted from tracker and domain."""
    # Given
    test_slots = [
        TextSlot("spam", mappings=[]),
        AnySlot("eggs", mappings=[]),
        BooleanSlot("ham", mappings=[]),
    ]
    test_events = [
        SlotSet("spam", "pemmican"),
        SlotSet("eggs", "scrambled"),
        SlotSet("ham", "true"),
    ]
    tracker = DialogueStateTracker.from_events(
        "test", evts=test_events, slots=test_slots
    )

    generator = CommandGenerator()
    # When
    document = generator._get_context_and_slots(tracker)
    # Then
    assert "context" in document
    for slot, event in zip(test_slots, test_events):
        assert slot.name in document["slots"]
        assert event.value in document["slots"][slot.name]


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
    generator = CommandGenerator()
    # When
    commands = generator._check_commands_against_startable_flows(
        commands, startable_flows
    )
    # Then
    assert commands == expected_commands


def test_command_processor_checks_flow_guards():
    """Test that the command processor checks flow guards."""
    # Given
    with patch(
        "rasa.dialogue_understanding.generator.command_generator.CommandGenerator._get_context_and_slots"
    ) as get_context_and_slots:
        with patch(
            "rasa.dialogue_understanding.generator.command_generator.CommandGenerator._check_commands_against_startable_flows"
        ) as check_commands_against_startable_flows:
            generator = WackyCommandGenerator()
            messages = [Message.build("What is your purpose?")]
            # When
            generator.process(messages, FlowsList([Flow("spam")]))
            # Then
            get_context_and_slots.assert_called_once()
            check_commands_against_startable_flows.assert_called_once()


def test_process_does_not_predict_commands_if_commands_already_present():
    """Test that predict_commands does not overwrite commands
    if commands are already set on message."""
    command_generator = CommandGenerator()

    command = StartFlowCommand("some flow").as_dict()

    test_message = Message.build(text="some message")
    test_message.set(COMMANDS, [command], add_to_output=True)

    assert len(test_message.get(COMMANDS)) == 1
    assert test_message.get(COMMANDS) == [command]

    mock_tracker = Mock()
    mock_tracker.get_slot = Mock(return_value=[])

    returned_message = command_generator.process(
        [test_message], flows=Mock(), tracker=mock_tracker
    )[0]

    assert len(returned_message.get(COMMANDS)) == 1
    assert returned_message.get(COMMANDS) == [command]
