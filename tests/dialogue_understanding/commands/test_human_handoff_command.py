from rasa.dialogue_understanding.commands.human_handoff_command import (
    HumanHandoffCommand,
)
from rasa.shared.core.events import UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.dialogue_understanding.commands.human_handoff_command import (
    HumanHandoffPatternFlowStackFrame,
)
from rasa.shared.core.events import DialogueStackUpdated


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

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, HumanHandoffPatternFlowStackFrame)
