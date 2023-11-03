from typing import Optional, List
from unittest.mock import Mock

import pytest

from rasa.dialogue_understanding.commands import Command, StartFlowCommand
from rasa.dialogue_understanding.generator.command_generator import CommandGenerator
from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.shared.core.flows import FlowsList
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
        generator.process([Message.build("test")], FlowsList(underlying_flows=[]))


def test_process_does_not_predict_commands_if_commands_already_present():
    """Test that predict_commands does not overwrite commands
    if commands are already set on message."""
    command_generator = CommandGenerator()

    command = StartFlowCommand("some flow").as_dict()

    test_message = Message.build(text="some message")
    test_message.set(COMMANDS, [command], add_to_output=True)

    assert len(test_message.get(COMMANDS)) == 1
    assert test_message.get(COMMANDS) == [command]

    returned_message = command_generator.process(
        [test_message], flows=Mock(), tracker=Mock()
    )[0]

    assert len(returned_message.get(COMMANDS)) == 1
    assert returned_message.get(COMMANDS) == [command]
