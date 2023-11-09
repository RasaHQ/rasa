from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.shared.core.events import DialogueStackUpdated, SlotSet, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker
import jsonpatch


def test_name_of_command():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ChitChatAnswerCommand.command() == "chitchat"


def test_from_dict():
    assert ChitChatAnswerCommand.from_dict({}) == ChitChatAnswerCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("hi", {"name": "greet"}),
        ],
    )
    command = ChitChatAnswerCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]

    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_chitchat"
