from rasa.dialogue_understanding.commands.error_command import ErrorCommand
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ErrorCommand.command() == "error"


def test_from_dict():
    assert ErrorCommand.from_dict({}) == ErrorCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = ErrorCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == "dialogue_stack"
    assert len(dialogue_stack_event.value) == 1

    frame = dialogue_stack_event.value[0]
    assert frame["type"] == "pattern_internal_error"
    assert frame["step_id"] == "START"
    assert frame["flow_id"] == "pattern_internal_error"
