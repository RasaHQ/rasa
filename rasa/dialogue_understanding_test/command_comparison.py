from typing import List

from rasa.dialogue_understanding.commands import ClarifyCommand, Command


def are_command_lists_equal(
    command_list_1: List[Command], command_list_2: List[Command]
) -> bool:
    """Check if the command lists are equal."""
    if len(command_list_1) != len(command_list_2):
        return False

    for command in command_list_1:
        if not is_command_present_in_list(command, command_list_2):
            return False

    return True


def is_command_present_in_list(
    command: Command,
    list_of_commands: List[Command],
) -> bool:
    """Check if the command is present in the list of commands."""
    for command_in_list in list_of_commands:
        if are_commands_equal(command, command_in_list):
            return True

    return False


def are_commands_equal(
    command_1: Command,
    command_2: Command,
) -> bool:
    """Compare the commands.

    Clarify commands are compared separately as options might be optional.
    """
    # as options are optional for clarify commands,
    # we need to check them separately
    if isinstance(command_1, ClarifyCommand) and isinstance(command_2, ClarifyCommand):
        return _are_clarify_commands_equal(command_1, command_2)

    # an exact match is required for all other commands
    return command_1 == command_2


def _are_clarify_commands_equal(
    expected_command: ClarifyCommand,
    predicted_command: ClarifyCommand,
) -> bool:
    # if the expected command contains options,
    # the predicted command should have the same options
    if expected_command.options:
        return sorted(expected_command.options) == sorted(predicted_command.options)

    # if the expected command does not contain options,
    # it does not matter whether the predicted command has options or not
    return True
