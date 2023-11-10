from rasa.dialogue_understanding.commands.human_handoff_command import (
    HumanHandoffCommand,
)
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert HumanHandoffCommand.command() == "human handoff"


def test_from_dict():
    assert HumanHandoffCommand.from_dict({}) == HumanHandoffCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = HumanHandoffCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []
