from rasa.dialogue_understanding.commands.error_command import ErrorCommand
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.core.events import DialogueStackUpdated, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
import jsonpatch


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
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, PatternFlowStackFrame)
    assert frame.type() == "pattern_internal_error"
    assert frame.step_id == "START"
    assert frame.flow_id == "pattern_internal_error"
