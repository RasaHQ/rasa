from rasa.dialogue_understanding.commands.change_flow_command import ChangeFlowCommand
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
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

    assert ChangeFlowCommand.regex_pattern() == r"^change"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()
