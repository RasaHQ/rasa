from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.human_handoff_command import (
    HumanHandoffCommand,
    HumanHandoffPatternFlowStackFrame,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.shared.core.events import DialogueStackUpdated, UserUttered
from rasa.shared.core.trackers import DialogueStateTracker


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


def test_to_dsl_default():
    command = HumanHandoffCommand()
    assert command.to_dsl() == "HumanHandoff()"


def test_from_dsl():
    assert HumanHandoffCommand.from_dsl(None) == HumanHandoffCommand()


def test_regex_pattern_default():
    assert HumanHandoffCommand.regex_pattern() == r"HumanHandoff\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = HumanHandoffCommand()
    assert command.to_dsl() == "hand over"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert HumanHandoffCommand.regex_pattern() == r"^hand over$"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(HumanHandoffCommand(), PromptCommand) is True
