from rasa.dialogue_understanding.commands.knowledge_answer_command import (
    KnowledgeAnswerCommand,
)
from rasa.shared.core.events import DialogueStackUpdated, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert KnowledgeAnswerCommand.command() == "knowledge"


def test_from_dict():
    assert KnowledgeAnswerCommand.from_dict({}) == KnowledgeAnswerCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = KnowledgeAnswerCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert frame.type() == "pattern_search"
