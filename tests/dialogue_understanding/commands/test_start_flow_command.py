import pytest
from rasa.dialogue_understanding.commands.start_flow_command import StartFlowCommand
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import SlotSet
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
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == DIALOGUE_STACK_SLOT

    dialogue_stack_dump = dialogue_stack_event.value
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 1
    assert dialogue_stack_dump[0]["frame_type"] == "regular"
    assert dialogue_stack_dump[0]["flow_id"] == "foo"
    assert dialogue_stack_dump[0]["step_id"] == "START"
    assert dialogue_stack_dump[0].get("frame_id") is not None


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
    tracker.update(
        SlotSet(
            DIALOGUE_STACK_SLOT,
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "START",
                    "frame_id": "test",
                }
            ],
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
    tracker.update(
        SlotSet(
            DIALOGUE_STACK_SLOT,
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "START",
                    "frame_id": "test",
                }
            ],
        )
    )
    command = StartFlowCommand(flow="bar")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == DIALOGUE_STACK_SLOT

    dialogue_stack_dump = dialogue_stack_event.value
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2
    assert dialogue_stack_dump[1]["frame_type"] == "interrupt"
    assert dialogue_stack_dump[1]["flow_id"] == "bar"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1].get("frame_id") is not None


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
    updated_tracker.update_with_events(events_bar, domain=None)
    events_foo = StartFlowCommand(flow="foo").run_command_on_tracker(
        updated_tracker, all_flows, tracker
    )

    assert len(events_foo) == 1

    dialogue_stack_event = events_foo[0]
    assert isinstance(dialogue_stack_event, SlotSet)
    assert dialogue_stack_event.key == DIALOGUE_STACK_SLOT

    dialogue_stack_dump = dialogue_stack_event.value
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    # both frames should be regular if they are started at the same time
    assert dialogue_stack_dump[1]["frame_type"] == "regular"
    assert dialogue_stack_dump[1]["flow_id"] == "foo"
    assert dialogue_stack_dump[0]["frame_type"] == "regular"
    assert dialogue_stack_dump[0]["flow_id"] == "bar"
