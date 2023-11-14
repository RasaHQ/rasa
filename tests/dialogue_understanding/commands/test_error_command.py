from rasa.dialogue_understanding.commands import ErrorCommand
from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.constants import RASA_PATTERN_INTERNAL_ERROR_DEFAULT
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ErrorCommand.command() == "error"


def test_default_values():
    command = ErrorCommand()
    assert command.error_type == RASA_PATTERN_INTERNAL_ERROR_DEFAULT
    assert isinstance(command.info, dict)
    assert len(command.info) == 0


def test_from_dict():
    assert ErrorCommand.from_dict({}) == ErrorCommand()


def test_from_dict_error_type():
    test_error_type = "test_error_type"
    test_info = {"info_a": "value_a", "info_b": "value_b"}
    test_data = {"error_type": test_error_type, "info": test_info}
    assert ErrorCommand.from_dict(test_data) == ErrorCommand(test_error_type, test_info)


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
    assert isinstance(frame, InternalErrorPatternFlowStackFrame)
    assert frame.type() == "pattern_internal_error"
    assert frame.step_id == "START"
    assert frame.flow_id == "pattern_internal_error"
    assert frame.error_type == RASA_PATTERN_INTERNAL_ERROR_DEFAULT
    assert frame.info == dict()
