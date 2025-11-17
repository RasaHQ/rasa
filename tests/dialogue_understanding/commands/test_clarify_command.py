import re

import jsonpatch
import pytest
import structlog.testing

from rasa.dialogue_understanding.commands.clarify_command import ClarifyCommand
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame
from rasa.dialogue_understanding.patterns.completed import (
    CompletedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
)
from rasa.engine.language import Language
from rasa.shared.core.events import (
    AgentInterrupted,
    DialogueStackUpdated,
    FlowCompleted,
)
from rasa.shared.core.slots import StrictCategoricalSlot
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import filter_logs, flows_from_str


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert ClarifyCommand.command() == "clarify"


def test_from_dict():
    assert ClarifyCommand.from_dict({"options": ["foo", "bar"]}) == ClarifyCommand(
        options=["foo", "bar"]
    )


def test_from_dict_fails_if_options_is_missing():
    with pytest.raises(ValueError):
        ClarifyCommand.from_dict({})


def test_run_command_ignores_non_existent_flows():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=["does-not-exist", "foo"])

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_clarification"
    assert frame["flow_id"] == "pattern_clarification"
    assert frame["step_id"] == "START"
    assert frame["names"] == ["foo"]
    assert frame["clarification_options"] == ""
    assert frame["clarification_ids"] == ["foo"]


def test_run_command_uses_name_of_flow():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            name: some foo
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=["foo"])

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_clarification"
    assert frame["names"] == ["some foo"]
    assert frame["clarification_ids"] == ["foo"]


def test_clarify_command_hash_equal():
    command = ClarifyCommand(options=["foo", "bar", "baz"])
    other_command = ClarifyCommand(options=["foo", "bar", "baz"])
    assert hash(command) == hash(other_command)
    assert command == other_command
    assert command.options == other_command.options


def test_clarify_command_hash_not_equal():
    command = ClarifyCommand(options=["foo", "bar", "buzz"])
    other_command = ClarifyCommand(options=["foo", "bar", "baz"])
    assert hash(command) != hash(other_command)
    assert command != other_command
    assert command.options != other_command.options


def test_to_dsl_default():
    command = ClarifyCommand(options=["foo", "bar", "baz"])
    assert command.to_dsl() == "Clarify(foo, bar, baz)"


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = ClarifyCommand(options=["foo", "bar", "baz"])
    assert command.to_dsl() == "disambiguate flows foo bar baz"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_default():
    assert ClarifyCommand.regex_pattern() == r"Clarify\(([\"\'a-zA-Z0-9_, -]*)\)"


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        ClarifyCommand.regex_pattern()
        == r"""^[\s\W\d]*disambiguate flows (["'a-zA-Z0-9_, -]*)[\W]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v1_with_special_characters():
    pattern = ClarifyCommand.regex_pattern()
    test_string = "Clarify('foo-bar', 'foo bar', 'foo_bar')"
    match = re.match(pattern, test_string)
    assert match and match.group(1) == "'foo-bar', 'foo bar', 'foo_bar'"


def test_regex_pattern_v2_with_special_characters():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    pattern = ClarifyCommand.regex_pattern()
    test_string = "disambiguate flows 'foo-bar', 'foo bar', 'foo_bar'"
    match = re.match(pattern, test_string)
    assert match and match.group(1) == "'foo-bar', 'foo bar', 'foo_bar'"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(ClarifyCommand([]), PromptCommand) is True


def test_clarify_command_uses_localized_flow_name(monkeypatch: pytest.MonkeyPatch):
    # Load a flow with translations.
    german_flow_name = "German foo"
    all_flows = flows_from_str(
        f"""
        flows:
          foo:
            description: flow foo
            name: foo flow
            translation:
                de:
                  name: {german_flow_name}
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a tracker with a language slot set to German language.
    language = Language.from_language_code("de", is_default=True)
    slots = [
        StrictCategoricalSlot(
            "language", [], initial_value=language.code, values=[language.code]
        )
    ]
    tracker = DialogueStateTracker.from_events("test", evts=[], slots=slots)

    # Run the clarify command.
    command = ClarifyCommand(options=["foo"])
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Check clarify command uses flow names with German translation.
    dialogue_stack_event = events[0]
    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert dialogue_stack_dump[0]["names"] == [german_flow_name]


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = ClarifyCommand(options=["foo", "bar", "baz"])
    assert command.to_dsl() == "disambiguate flows foo bar baz"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        ClarifyCommand.regex_pattern()
        == r"""^[\s\W\d]*disambiguate flows (["'a-zA-Z0-9_, -]*)[\W]*$"""
    )

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_run_command_on_tracker_interrupts_agent_and_adds_event():
    tracker = DialogueStateTracker.from_events("test", evts=[])

    agent_frame = AgentStackFrame(
        frame_id="agent-frame",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="car_research",
    )
    tracker.update_stack(DialogueStack(frames=[agent_frame]))

    all_flows = flows_from_str(
        """
        flows:
          flow-a:
            description: flow a
            steps:
            - id: first_step
              action: action_listen
          flow-b:
            description: flow b
            steps:
            - id: first_step
              action: action_listen
        """
    )
    original_tracker = tracker

    command = ClarifyCommand(options=["flow-a", "flow-b"])
    events = command.run_command_on_tracker(tracker, all_flows, original_tracker)

    # Check that an AgentInterrupted event is created
    assert any(
        isinstance(e, AgentInterrupted)
        and e.agent_id == "car-research"
        and e.flow_id == "car_research"
        for e in events
        if isinstance(e, AgentInterrupted)
    )


def test_run_command_handles_empty_clarification_options():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = ClarifyCommand(options=[])

    with structlog.testing.capture_logs() as caplog:
        events = command.run_command_on_tracker(tracker, all_flows, tracker)

        logs = filter_logs(
            caplog,
            "clarify_command.empty_clarification",
            "debug",
        )
        assert len(logs) == 1
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())
    assert len(dialogue_stack_dump) == 1

    frame = dialogue_stack_dump[0]
    assert frame["type"] == "pattern_clarification"
    assert frame["flow_id"] == "pattern_clarification"
    assert frame["step_id"] == "START"
    assert frame["names"] == []
    assert frame["clarification_options"] == ""


def test_run_command_on_tracker_removes_pattern_completed_frames():
    tracker = DialogueStateTracker.from_events("test", evts=[])

    all_flows = flows_from_str(
        """
        flows:
          flow-a:
            description: flow a
            steps:
            - id: first_step
              action: action_listen
          flow-b:
            description: flow b
            steps:
            - id: first_step
              action: action_listen
        """
    )

    pattern_completed_frame = CompletedPatternFlowStackFrame(
        frame_id="pattern-completed-frame",
    )
    tracker.update_stack(DialogueStack(frames=[pattern_completed_frame]))

    command = ClarifyCommand(options=["flow-a", "flow-b"])
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    assert len(events) == 2
    assert isinstance(events[0], FlowCompleted)
    assert events[0].flow_id == "pattern_completed"
    assert isinstance(events[1], DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(events[1].update)
    assert len(updated_stack.frames) == 1
    assert isinstance(updated_stack.frames[0], ClarifyPatternFlowStackFrame)
