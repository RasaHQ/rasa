from rasa.dialogue_understanding.commands.can_not_handle_command import (
    CannotHandleCommand,
)
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CannotHandleCommand.command() == "cannot handle"


def test_from_dict():
    assert CannotHandleCommand.from_dict({}) == CannotHandleCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = CannotHandleCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []
