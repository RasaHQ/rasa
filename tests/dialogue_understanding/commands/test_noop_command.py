from rasa.dialogue_understanding.commands.noop_command import NoopCommand
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert NoopCommand.command() == "noop"


def test_from_dict():
    assert NoopCommand.from_dict({}) == NoopCommand()


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = NoopCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_is_not_instance_of_prompt_command():
    # Check if the command does not adhere to the PromptCommand protocol.
    assert isinstance(NoopCommand(), PromptCommand) is False
