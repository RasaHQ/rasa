from typing import Optional, List

import pytest

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.generator.command_generator import CommandGenerator
from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.shared.core.flows.flow import FlowsList
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
    generator.process(messages, FlowsList([]))
    commands = [m.get(COMMANDS) for m in messages]

    assert len(commands[0]) == 0
    assert len(commands[1]) == 1
    assert commands[1][0]["command"] == ChitChatAnswerCommand.command()


def test_command_generator_still_throws_not_implemented_error():
    # This test can be removed if the predict_commands method stops to be abstract
    generator = CommandGenerator()
    with pytest.raises(NotImplementedError):
        generator.process([Message.build("test")], FlowsList([]))
