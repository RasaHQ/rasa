from rasa.dialogue_understanding.commands.knowledge_answer_command import (
    KnowledgeAnswerCommand,
)
from rasa.shared.core.events import SlotSet, UserUttered
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
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == "dialogue_stack"
    assert len(dialogue_stack_event.value) == 1

    frame = dialogue_stack_event.value[0]
    assert frame["type"] == "search"
