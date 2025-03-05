import jsonpatch

from rasa.dialogue_understanding.commands.chit_chat_answer_command import (
    ChitChatAnswerCommand,
)
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.shared.core.events import DialogueStackUpdated, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


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


def test_to_dsl_default():
    command = ChitChatAnswerCommand()
    assert command.to_dsl() == "ChitChat()"


def test_from_dsl():
    assert ChitChatAnswerCommand.from_dsl(None) == ChitChatAnswerCommand()


def test_regex_pattern_default():
    assert ChitChatAnswerCommand.regex_pattern() == r"ChitChat\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = ChitChatAnswerCommand()
    assert command.to_dsl() == "offtopic reply"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert ChitChatAnswerCommand.regex_pattern() == r"^[^\w]*offtopic reply$"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(ChitChatAnswerCommand(), PromptCommand) is True
