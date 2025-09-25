import re

import pytest

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.continue_agent_command import (
    ContinueAgentCommand,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ContinueAgentCommand.command() == "continue agent"


def test_from_dict():
    assert ContinueAgentCommand.from_dict({}) == ContinueAgentCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ContinueAgentCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_to_dsl_default():
    command = ContinueAgentCommand()
    assert command.to_dsl() == "ContinueAgent()"


def test_from_dsl():
    assert ContinueAgentCommand.from_dsl(None) == ContinueAgentCommand()


def test_regex_pattern_default():
    assert ContinueAgentCommand.regex_pattern() == r"ContinueAgent\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = ContinueAgentCommand()
    assert command.to_dsl() == "continue agent"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        ContinueAgentCommand.regex_pattern() == r"""^[\s\W\d]*continue agent['"`]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(ContinueAgentCommand(), PromptCommand) is True


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = ContinueAgentCommand()
    assert command.to_dsl() == "continue agent"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        ContinueAgentCommand.regex_pattern() == r"""^[\s\W\d]*continue agent['"`]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


@pytest.mark.parametrize(
    "input_action",
    [
        "continue agent",
        " - continue agent",
        " --> continue agent",
        " *** continue agent",
        " 1. continue agent",
        " 2) continue agent",
        "'continue agent'",
        "```continue agent```",
        "```plaintext\ncontinue agent```",
        "```plaintext\n'continue agent'```",
    ],
)
def test_parse(input_action):
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    pattern = re.compile(ContinueAgentCommand.regex_pattern())
    match = pattern.search(input_action)
    parsed_command = ContinueAgentCommand.from_dsl(match)
    assert parsed_command == ContinueAgentCommand()

    CommandSyntaxManager.reset_syntax_version()
