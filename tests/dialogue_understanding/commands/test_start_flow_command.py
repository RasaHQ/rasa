import pytest
from rasa.dialogue_understanding.commands.start_flow_command import StartFlowCommand
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.shared.core.events import DialogueStackUpdated, FlowInterrupted
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


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
            steps:
            - id: first_step
              action: action_listen
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = StartFlowCommand(flow="pattern_foo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 0


def test_run_start_flow_interrupting_existing_flow():
    all_flows = flows_from_str(
        """
        flows:
          foo:
            steps:
            - id: first_step
              action: action_listen
          bar:
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
            steps:
            - id: first_step
              action: action_listen
          bar:
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
