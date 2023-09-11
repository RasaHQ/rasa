from rasa.cdu.commands.can_not_handle_command import CanNotHandleCommand
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CanNotHandleCommand.command() == "cant handle"


def test_from_dict():
    assert CanNotHandleCommand.from_dict({}) == CanNotHandleCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = CanNotHandleCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []
