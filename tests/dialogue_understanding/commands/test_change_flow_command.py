from rasa.dialogue_understanding.commands.change_flow_command import ChangeFlowCommand
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    assert ChangeFlowCommand.command() == "change_flow"


def test_from_dict():
    assert ChangeFlowCommand.from_dict({}) == ChangeFlowCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ChangeFlowCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_to_dsl():
    command = ChangeFlowCommand()
    assert command.to_dsl() == "ChangeFlow()"


def test_from_dsl():
    assert ChangeFlowCommand.from_dsl(None) == ChangeFlowCommand()


def test_regex_pattern():
    assert ChangeFlowCommand.regex_pattern() == r"ChangeFlow\(\)"
