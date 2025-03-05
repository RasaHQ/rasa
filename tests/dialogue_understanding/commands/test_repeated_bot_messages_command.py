from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.commands.repeat_bot_messages_command import (
    RepeatBotMessagesCommand,
)
from rasa.dialogue_understanding.patterns.repeat import (
    RepeatBotMessagesPatternFlowStackFrame,
)
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker


def test_command_name():
    assert RepeatBotMessagesCommand.command() == "repeat"


def test_from_dict():
    assert RepeatBotMessagesCommand.from_dict({}) == RepeatBotMessagesCommand()


def test_run_command_on_tracker():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = RepeatBotMessagesCommand()

    events = command.run_command_on_tracker(tracker, [], tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, RepeatBotMessagesPatternFlowStackFrame)


def test_to_dsl_default():
    command = RepeatBotMessagesCommand()
    assert command.to_dsl() == "RepeatLastBotMessages()"


def test_from_dsl():
    assert RepeatBotMessagesCommand.from_dsl(None) == RepeatBotMessagesCommand()


def test_regex_pattern_default():
    assert RepeatBotMessagesCommand.regex_pattern() == r"RepeatLastBotMessages\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the new DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = RepeatBotMessagesCommand()
    assert command.to_dsl() == "repeat message"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert RepeatBotMessagesCommand.regex_pattern() == r"^[^\w]*repeat message$"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(RepeatBotMessagesCommand(), PromptCommand) is True
