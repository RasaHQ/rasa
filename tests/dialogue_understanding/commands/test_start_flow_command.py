import re
from typing import cast

import pytest

from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.prompt_command import PromptCommand
from rasa.dialogue_understanding.commands.start_flow_command import StartFlowCommand
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.events import (
    AgentInterrupted,
    AgentResumed,
    DialogueStackUpdated,
    FlowInterrupted,
    FlowResumed,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert StartFlowCommand.command() == "start flow"


def test_from_dict():
    assert StartFlowCommand.from_dict({"flow": "test"}) == StartFlowCommand(flow="test")


def test_from_dict_fails_if_parameter_is_missing():
    with pytest.raises(ValueError):
        StartFlowCommand.from_dict({})


def test_run_command_on_tracker():
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
    command = StartFlowCommand(flow="foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.frame_type == "regular"
    assert frame.flow_id == "foo"
    assert frame.step_id == "START"
    assert frame.frame_id is not None


def test_run_start_flow_that_does_not_exist():
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
    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 0


def test_run_start_flow_that_is_already_on_the_stack():
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
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "START",
                    "frame_id": "test",
                }
            ]
        )
    )
    command = StartFlowCommand(flow="foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 0


def test_run_start_flow_which_is_a_pattern():
    all_flows = flows_from_str(
        """
        flows:
          pattern_foo:
            description: pattern foo
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = StartFlowCommand(flow="pattern_foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1


def test_run_start_flow_interrupting_existing_flow():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "START",
                    "frame_id": "test",
                }
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 2

    # the first event should be a flow interrupted event
    flow_started_event = events[0]
    assert flow_started_event == FlowInterrupted(flow_id="foo", step_id="START")

    dialogue_stack_event = events[1]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.frame_type == "interrupt"
    assert frame.flow_id == "bar"
    assert frame.step_id == "START"
    assert frame.frame_id is not None


def test_run_start_flow_with_multiple_flows():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])

    events_bar = StartFlowCommand(flow="bar").run_command_on_tracker(
        tracker, all_flows, tracker
    )

    updated_tracker = tracker.copy()
    updated_tracker.update_with_events(events_bar)
    events_foo = StartFlowCommand(flow="foo").run_command_on_tracker(
        updated_tracker, all_flows, tracker
    )

    assert len(events_foo) == 1

    dialogue_stack_event = events_foo[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = updated_tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    # both frames should be regular if they are started at the same time
    assert isinstance(updated_stack.frames[1], UserFlowStackFrame)
    assert updated_stack.frames[1].frame_type == "regular"
    assert updated_stack.frames[1].flow_id == "foo"
    assert isinstance(updated_stack.frames[0], UserFlowStackFrame)
    assert updated_stack.frames[0].frame_type == "regular"
    assert updated_stack.frames[0].flow_id == "bar"


def test_run_start_flow_resume_existing_flow_simple():
    """Test resuming a flow that's already on the stack with a simple configuration."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
            - id: second_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a stack with foo on top, bar below
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check events
    assert events[2] == FlowInterrupted(flow_id="foo", step_id="first_step")
    assert events[0] == FlowResumed(flow_id="bar", step_id="first_step")

    # Check stack reordering
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    # bar should now be on top, foo below
    assert len(updated_stack.frames) == 2
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].step_id == "first_step"
    assert updated_stack.frames[0].frame_type == "regular"
    assert updated_stack.frames[1].flow_id == "bar"
    assert updated_stack.frames[1].step_id == "first_step"
    assert updated_stack.frames[1].frame_type == "interrupt"


def test_run_start_flow_resume_existing_flow_with_pattern_frames():
    """Test resuming a flow with pattern frames in between."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a stack with pattern frames between flow frames
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "pattern_collect_information",
                    "flow_id": "pattern_collect_information",
                    "step_id": "START",
                    "frame_id": "pattern-frame",
                    "collect": "first_step",
                    "collect_action": "action_ask_fist_step",
                    "utter": "utter_ask_fist_step",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check stack reordering - bar and pattern should move to top
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 3
    # foo should be at bottom
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].frame_type == "regular"
    # bar and pattern should be on top
    assert updated_stack.frames[1].flow_id == "bar"
    assert updated_stack.frames[1].frame_type == "interrupt"
    assert updated_stack.frames[2].flow_id == "pattern_collect_information"


def test_run_start_flow_resume_existing_flow_with_multiple_flows():
    """Test resuming a flow when there are multiple flows on the stack."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
          baz:
            description: flow baz
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a stack with three flows: baz, bar, foo (top to bottom)
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "baz",
                    "step_id": "first_step",
                    "frame_id": "baz-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check stack reordering
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 3
    # foo should be at bottom
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].frame_type == "regular"
    # bar and baz should be on top
    assert updated_stack.frames[1].flow_id == "baz"
    assert updated_stack.frames[1].frame_type == "interrupt"
    assert updated_stack.frames[2].flow_id == "bar"
    assert updated_stack.frames[2].frame_type == "interrupt"


def test_run_start_flow_resume_existing_flow_with_interrupt_frames():
    """Test resuming a flow that has interrupt frames."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a stack with bar having an interrupt frame
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check stack reordering
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2
    # foo should be at bottom
    assert updated_stack.frames[0].flow_id == "foo"
    # bar should be on top, still as interrupt
    assert updated_stack.frames[1].flow_id == "bar"
    assert updated_stack.frames[1].frame_type == "interrupt"


def test_run_start_flow_resume_existing_flow_preserves_frame_ids():
    """Test that frame IDs are preserved when resuming flows."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # Create a stack with specific frame IDs
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-original-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-original-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check that frame IDs are preserved
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2
    assert updated_stack.frames[0].frame_id == "foo-original-frame"
    assert updated_stack.frames[1].frame_id == "bar-original-frame"


def test_run_start_flow_resume_existing_flow_empty_stack():
    """Test resuming a flow when stack is empty."""
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
    command = StartFlowCommand(flow="foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1  # Just the stack update event

    dialogue_stack_event = events[0]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].frame_type == "regular"


def test_run_start_flow_resume_existing_flow_with_multiple_flows_and_called_flows():
    """Test resuming a flow with multiple flows incl a called flow on the stack."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
            - id: call_step
              call: foo_bar
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
          baz:
            description: flow baz
            steps:
            - id: first_step
              action: action_listen
          foo_bar:
            description: flow foo_bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # stack with four flows: bar, foo, foo_bar (called flow), baz
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "foo",
                    "step_id": "call_step",
                    "frame_id": "foo-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "call",
                    "flow_id": "foo_bar",
                    "step_id": "first_step",
                    "frame_id": "foo_bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "baz",
                    "step_id": "first_step",
                    "frame_id": "baz-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check stack reordering
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 4
    # foo should be at bottom
    assert updated_stack.frames[0].flow_id == "bar"
    assert updated_stack.frames[0].frame_type == "regular"
    # bar and baz should be on top
    assert updated_stack.frames[1].flow_id == "baz"
    assert updated_stack.frames[1].frame_type == "interrupt"
    assert updated_stack.frames[2].flow_id == "foo"
    assert updated_stack.frames[2].frame_type == "interrupt"
    assert updated_stack.frames[3].flow_id == "foo_bar"
    assert updated_stack.frames[3].frame_type == "call"


def test_run_start_flow_resume_existing_flow_with_multiple_flows_and_linked_flows():
    """Test resuming a flow with multiple flows, incl a linked flow on the stack."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
            - id: link_step
              link: foo_bar
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
          baz:
            description: flow baz
            steps:
            - id: first_step
              action: action_listen
          foo_bar:
            description: flow foo_bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    # stack with four flows: bar, foo, foo_bar (called flow), baz
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "foo",
                    "step_id": "link_step",
                    "frame_id": "foo-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "link",
                    "flow_id": "foo_bar",
                    "step_id": "first_step",
                    "frame_id": "foo_bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "baz",
                    "step_id": "first_step",
                    "frame_id": "baz-frame",
                },
            ]
        )
    )

    command = StartFlowCommand(flow="foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check stack reordering
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 4
    # foo should be at bottom
    assert updated_stack.frames[0].flow_id == "bar"
    assert updated_stack.frames[0].frame_type == "regular"
    # bar and baz should be on top
    assert updated_stack.frames[1].flow_id == "baz"
    assert updated_stack.frames[1].frame_type == "interrupt"
    assert updated_stack.frames[2].flow_id == "foo"
    assert updated_stack.frames[2].frame_type == "interrupt"
    assert updated_stack.frames[3].flow_id == "foo_bar"
    assert updated_stack.frames[3].frame_type == "link"


def test_run_start_flow_resume_existing_flow_with_agent():
    """Test resuming a flow and its agent that were interrupted."""
    from rasa.shared.core.events import FlowInterrupted, FlowResumed

    all_flows = flows_from_str(
        """
        flows:
          car_research:
            description: flow with agent
            steps:
            - id: first_step
              action: action_listen
          other_flow:
            description: another flow
            steps:
            - id: first_step
              action: action_listen
        """
    )
    tracker = DialogueStateTracker.from_events("test", evts=[])
    # Start car_research (with agent)
    agent_frame = AgentStackFrame(
        frame_id="agent-frame",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="car_research",
    )
    user_frame = UserFlowStackFrame(
        flow_id="car_research",
        step_id="first_step",
        frame_id="car_research-frame",
        frame_type=FlowStackFrameType.REGULAR,
    )
    tracker.update_stack(DialogueStack(frames=[user_frame, agent_frame]))

    # Interrupt car_research by starting other_flow
    interrupt_command = StartFlowCommand(flow="other_flow")
    events = interrupt_command.run_command_on_tracker(tracker, all_flows, tracker)
    tracker.update_with_events(events)

    # Now car_research and its agent are interrupted, other_flow is on top
    # Resume car_research
    resume_command = StartFlowCommand(flow="car_research")
    events = resume_command.run_command_on_tracker(tracker, all_flows, tracker)

    # Assert AgentResumed and FlowResumed are present for the correct agent/flow
    assert any(
        isinstance(event, AgentResumed) and event.agent_id == "car-research"
        for event in events
    )
    assert any(
        isinstance(event, FlowResumed) and event.flow_id == "car_research"
        for event in events
    )
    # There should also be a FlowInterrupted event for the interrupted flow
    assert any(
        isinstance(event, FlowInterrupted) and event.flow_id == "other_flow"
        for event in events
    )


def test_to_dsl_default():
    command = StartFlowCommand("foo")
    assert command.to_dsl() == "StartFlow(foo)"


def test_regex_pattern_default():
    assert (
        StartFlowCommand.regex_pattern() == r"StartFlow\(['\"]?([a-zA-Z0-9_-]+)['\"]?\)"
    )


def test_to_dsl_v2_command_syntax():
    # Set the syntax version to v2 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    command = StartFlowCommand("foo")
    assert command.to_dsl() == "start flow foo"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v2_command_syntax():
    # Set the syntax version to v2 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v2)

    assert (
        StartFlowCommand.regex_pattern()
        == r"""^[\s\W\d]*start flow ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_from_dsl():
    action = "StartFlow(foo)"
    pattern = re.compile(StartFlowCommand.regex_pattern())
    match = pattern.search(action)
    assert StartFlowCommand.from_dsl(match) == StartFlowCommand("foo")


def test_is_instance_of_prompt_command():
    # Check if the command adheres to the PromptCommand protocol.
    assert isinstance(StartFlowCommand([]), PromptCommand) is True


def test_to_dsl_v3_command_syntax():
    # Set the syntax version to v3 to test the DSL.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    command = StartFlowCommand("foo")
    assert command.to_dsl() == "start flow foo"

    # Reset the syntax version to default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_regex_pattern_v3_command_syntax():
    # Set the syntax version to v3 to test the new regex pattern.
    CommandSyntaxManager.set_syntax_version(CommandSyntaxVersion.v3)

    assert (
        StartFlowCommand.regex_pattern()
        == r"""^[\s\W\d]*start flow ['"`]?([a-zA-Z0-9_-]+)['"`]*"""
    )

    # Reset the syntax version to the default, otherwise it will affect other tests.
    CommandSyntaxManager.reset_syntax_version()


def test_start_flow_interrupts_active_agent_frame():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    user_stack_frame = UserFlowStackFrame(
        flow_id="foo", step_id="START", frame_id="some-frame-id"
    )
    agent_frame = AgentStackFrame(
        frame_id="agent-frame",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="car-research",
        flow_id="foo",
    )
    tracker.update_stack(DialogueStack(frames=[user_stack_frame, agent_frame]))

    stack_before = tracker.stack

    command = StartFlowCommand(flow="bar")

    # WHEN: starting a new flow while an agent awaits input
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Assert AgentInterrupted is present for the correct agent
    assert any(
        isinstance(event, AgentInterrupted) and event.agent_id == "car-research"
        for event in events
    )

    for event in events:
        if isinstance(event, DialogueStackUpdated):
            updated_stack = stack_before.update_from_patch(event.update)
            assert len(updated_stack.frames) == 3

            assert isinstance(updated_stack.frames[1], AgentStackFrame)
            assert (
                cast(AgentStackFrame, updated_stack.frames[1]).frame_id == "agent-frame"
            )
            assert (
                cast(AgentStackFrame, updated_stack.frames[1]).state
                == AgentState.INTERRUPTED
            )

            assert isinstance(updated_stack.frames[2], UserFlowStackFrame)
            assert cast(UserFlowStackFrame, updated_stack.frames[2]).flow_id == "bar"
            assert (
                cast(UserFlowStackFrame, updated_stack.frames[2]).frame_type
                == FlowStackFrameType.INTERRUPT
            )


def test_start_flow_removes_continue_interrupted_frames_when_same_flow():
    """Test that starting the same flow removes continue interrupted frames."""
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

    # Create a stack with foo flow and continue interrupted pattern frame on top
    user_stack_frame = UserFlowStackFrame(
        flow_id="foo", step_id="START", frame_id="foo-frame-id"
    )
    continue_interrupted_frame = ContinueInterruptedPatternFlowStackFrame(
        frame_id="continue-pattern-frame",
        step_id="continue_step",
        interrupted_flow_names=["previous_flow"],
        interrupted_flow_ids=["previous_flow_id"],
        interrupted_flow_options="previous_flow_options",
    )

    tracker.update_stack(
        DialogueStack(frames=[user_stack_frame, continue_interrupted_frame])
    )

    # Verify the stack has both frames initially
    assert len(tracker.stack.frames) == 2
    assert tracker.stack.top() == continue_interrupted_frame

    command = StartFlowCommand(flow="foo")

    # WHEN: starting the same flow that's already on top
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Should return a dialogue stack updated event
    assert len(events) == 1
    assert isinstance(events[0], DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(events[0].update)
    assert len(updated_stack.frames) == 1
    assert updated_stack.top() == user_stack_frame


def test_start_flow_resumes_flow_and_removes_continue_interrupted_frames():
    """Test that resuming a flow removes continue interrupted frames."""
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])

    # Create a stack with bar flow on bottom, foo flow on top, and continue interrupted
    # pattern frame
    bar_stack_frame = UserFlowStackFrame(
        flow_id="bar", step_id="first_step", frame_id="bar-frame-id"
    )
    foo_stack_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="foo-frame-id"
    )
    continue_interrupted_frame = ContinueInterruptedPatternFlowStackFrame(
        frame_id="continue-pattern-frame",
        step_id="continue_step",
        interrupted_flow_names=["previous_flow"],
        interrupted_flow_ids=["previous_flow_id"],
        interrupted_flow_options="previous_flow_options",
    )

    tracker.update_stack(
        DialogueStack(
            frames=[bar_stack_frame, foo_stack_frame, continue_interrupted_frame]
        )
    )

    # Verify the stack has all three frames initially
    assert len(tracker.stack.frames) == 3
    assert tracker.stack.top() == continue_interrupted_frame

    command = StartFlowCommand(flow="bar")

    # WHEN: resuming the bar flow that's below the foo flow
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Should return events for resuming the flow
    assert len(events) == 3  # FlowResumed, DialogueStackUpdated, FlowInterrupted

    # Check that the continue interrupted frame is removed and bar flow is resumed
    dialogue_stack_event = events[1]
    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    # Should have 2 frames: bar (resumed) and foo (interrupted)
    assert len(updated_stack.frames) == 2
    # the bar_stack_frame should be on top with frame type "interrupt"
    assert updated_stack.top().flow_id == "bar"
    assert updated_stack.top().frame_type == FlowStackFrameType.INTERRUPT
    # the foo_stack_frame should be on bottom with frame type "regular"
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].frame_type == FlowStackFrameType.REGULAR


def test_start_flow_removes_pattern_continue_interrupted_frames_when_active():
    """Test that pattern continue interrupted frames are removed when they are
    currently active and we start a flow.
    """
    all_flows = flows_from_str(
        """
        flows:
          foo:
            description: flow foo
            steps:
            - id: first_step
              action: action_listen
          bar:
            description: flow bar
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])

    # Create a stack where the pattern continue interrupted frame is at the top (active)
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="START", frame_id="foo-frame"
    )
    pattern_frame = ContinueInterruptedPatternFlowStackFrame(
        frame_id="pattern-frame",
        step_id="pattern_step",
        interrupted_flow_names=["interrupted_flow"],
        interrupted_flow_ids=["interrupted_flow_id"],
        interrupted_flow_options="interrupted_flow_options",
    )

    # Stack with pattern frame at the top (currently active)
    tracker.update_stack(DialogueStack(frames=[user_frame, pattern_frame]))

    # Verify the pattern frame is currently active (at the top)
    assert tracker.stack.top() == pattern_frame
    assert isinstance(tracker.stack.top(), ContinueInterruptedPatternFlowStackFrame)

    command = StartFlowCommand(flow="bar")

    # WHEN: starting a new flow while pattern continue interrupted frame is active
    events = command.run_command_on_tracker(tracker, all_flows, tracker)

    # Should return events for starting the new flow
    assert len(events) == 2  # FlowInterrupted, DialogueStackUpdated

    # Check that the pattern frame has been removed and new flow is started
    dialogue_stack_event = events[1]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    # Should have 2 frames: foo (interrupted) and bar (new flow)
    assert len(updated_stack.frames) == 2
    # The pattern frame should be removed
    assert not any(
        isinstance(frame, ContinueInterruptedPatternFlowStackFrame)
        for frame in updated_stack.frames
    )
    # The new bar flow should be on top
    assert updated_stack.top().flow_id == "bar"
    assert updated_stack.top().frame_type == FlowStackFrameType.INTERRUPT
    # The original foo flow should be below
    assert updated_stack.frames[0].flow_id == "foo"
    assert updated_stack.frames[0].frame_type == FlowStackFrameType.REGULAR
