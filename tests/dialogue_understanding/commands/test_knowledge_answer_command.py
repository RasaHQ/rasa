from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.knowledge_answer_command import (
    KnowledgeAnswerCommand,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
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


def test_to_dsl_default():
    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "SearchAndReply()"


def test_from_dsl():
    assert KnowledgeAnswerCommand.from_dsl(None) == KnowledgeAnswerCommand()


def test_regex_pattern_default():
    assert KnowledgeAnswerCommand.regex_pattern() == r"SearchAndReply\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "provide info"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        KnowledgeAnswerCommand.regex_pattern() == r"""^[\s\W\d]*provide info['"`]*$"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(KnowledgeAnswerCommand(), PromptCommand) is True


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = KnowledgeAnswerCommand()
    assert command.to_dsl() == "search and reply"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        KnowledgeAnswerCommand.regex_pattern()
        == r"""^[\s\W\d]*search and reply['"`]*$"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()
