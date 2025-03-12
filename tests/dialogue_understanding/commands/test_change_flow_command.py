import re

import pytest

from rasa.dialogue_understanding.commands.change_flow_command import ChangeFlowCommand
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    assert ChangeFlowCommand.command() == "change_flow"


def test_from_dict():
    assert ChangeFlowCommand.from_dict({}) == ChangeFlowCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ChangeFlowCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_to_dsl_default():
    command = ChangeFlowCommand()
    assert command.to_dsl() == "ChangeFlow()"


def test_from_dsl():
    assert ChangeFlowCommand.from_dsl(None) == ChangeFlowCommand()


def test_regex_pattern_default():
    assert ChangeFlowCommand.regex_pattern() == r"ChangeFlow\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the new DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = ChangeFlowCommand()
    assert command.to_dsl() == "change"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert ChangeFlowCommand.regex_pattern() == r"""^[\s\W\d]*change['"`]*$"""

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(ChangeFlowCommand(), PromptCommand) is True


@pytest.mark.parametrize(
    "input_action",
    [
        "change",
        " - change",
        " --> change",
        " *** change",
        " 1. change",
        " 2) change",
        "'change'",
        "```change```",
        "```plaintext\nchange```",
        "```plaintext\n'change'```",
    ],
)
def test_parse(input_action):
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    pattern = re.compile(ChangeFlowCommand.regex_pattern())
    match = pattern.search(input_action)
    parsed_command = ChangeFlowCommand.from_dsl(match)
    assert parsed_command == ChangeFlowCommand()

    CommandSyntaxManager.reset_syntax_version()
