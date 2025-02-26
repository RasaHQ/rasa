from rasa.dialogue_understanding.commands.can_not_handle_command import (
    CannotHandleCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.patterns.cannot_handle import (
    FLOW_PATTERN_CANNOT_HANDLE,
    CannotHandlePatternFlowStackFrame,
)
from rasa.shared.constants import RASA_PATTERN_CANNOT_HANDLE_DEFAULT
from rasa.shared.core.events import DialogueStackUpdated, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CannotHandleCommand.command() == "cannot handle"


def test_default_values():
    command = CannotHandleCommand()
    assert command.reason == RASA_PATTERN_CANNOT_HANDLE_DEFAULT


def test_from_dict():
    assert CannotHandleCommand.from_dict({}) == CannotHandleCommand()


def test_from_dict_reason():
    test_data = {"reason": "test_reason"}
    assert CannotHandleCommand.from_dict(test_data) == CannotHandleCommand(
        "test_reason"
    )


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = CannotHandleCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, CannotHandlePatternFlowStackFrame)
    assert frame.type() == FLOW_PATTERN_CANNOT_HANDLE
    assert frame.step_id == "START"


def test_to_dsl_default():
    command = CannotHandleCommand("test_reason")
    assert command.to_dsl() == "CannotHandle()"


def test_from_dsl():
    assert CannotHandleCommand.from_dsl(None) == CannotHandleCommand()


def test_regex_pattern_default():
    assert CannotHandleCommand.regex_pattern() == r"CannotHandle\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = CannotHandleCommand("test_reason")
    assert command.to_dsl() == "cannot handle"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert CannotHandleCommand.regex_pattern() == r"^cannot handle$"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()
