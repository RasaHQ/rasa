from rasa.dialogue_understanding.commands import SessionStartCommand
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker
import jsonpatch


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert SessionStartCommand.command() == "session start"


def test_from_dict():
    assert SessionStartCommand.from_dict({}) == SessionStartCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = SessionStartCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]

    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_session_start"
