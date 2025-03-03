from rasa.dialogue_understanding.commands import SkipQuestionCommand
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.patterns.skip_question import (
    FLOW_PATTERN_SKIP_QUESTION,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert SkipQuestionCommand.command() == "skip question"


def test_from_dict():
    assert SkipQuestionCommand.from_dict({}) == SkipQuestionCommand()


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = SkipQuestionCommand()

    assert command.run_command_on_tracker(tracker, [], tracker) == []


def test_run_command_on_tracker():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
                description: test my flow
                steps:
                - id: collect_foo
                  collect: foo
                  next: collect_bar
                - id: collect_bar
                  collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                },
            ]
        )
    )
    command = SkipQuestionCommand()

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, PatternFlowStackFrame)
    assert frame.type() == FLOW_PATTERN_SKIP_QUESTION
    assert frame.step_id == "START"


def test_to_dsl_default():
    command = SkipQuestionCommand()
    assert command.to_dsl() == "SkipQuestion()"


def test_from_dsl():
    assert SkipQuestionCommand.from_dsl(None) == SkipQuestionCommand()


def test_regex_pattern_default():
    assert SkipQuestionCommand.regex_pattern() == r"SkipQuestion\(\)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = SkipQuestionCommand()
    assert command.to_dsl() == "skip"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert SkipQuestionCommand.regex_pattern() == r"^skip$"

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(SkipQuestionCommand(), PromptCommand) is True
