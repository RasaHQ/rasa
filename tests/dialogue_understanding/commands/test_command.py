import pytest
from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotCommand


def test_command_from_json():
    data = {"command": "set slot", "name": "foo", "value": "bar"}
    assert Command.command_from_json(data) == SetSlotCommand(name="foo", value="bar")


def test_command_from_dict_handles_unknown_commands():
    data = {"command": "unknown"}
    with pytest.raises(ValueError):
        Command.command_from_json(data)


def test_command_as_dict():
    command = SetSlotCommand(name="foo", value="bar")
    assert command.as_dict() == {"command": "set slot", "name": "foo", "value": "bar"}
