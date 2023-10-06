import pytest
from rasa.dialogue_understanding.commands.set_slot_command import SetSlotCommand
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert SetSlotCommand.command() == "set slot"


def test_from_dict():
    assert SetSlotCommand.from_dict({"name": "foo", "value": "bar"}) == SetSlotCommand(
        name="foo", value="bar"
    )


def test_from_dict_fails_if_no_parameters():
    with pytest.raises(ValueError):
        SetSlotCommand.from_dict({})


def test_from_dict_fails_if_value_is_missing():
    with pytest.raises(ValueError):
        SetSlotCommand.from_dict({"name": "bar"})


def test_from_dict_fails_if_name_is_missing():
    with pytest.raises(ValueError):
        SetSlotCommand.from_dict({"value": "foo"})


def test_run_command_skips_if_slot_is_set_to_same_value():
    tracker = DialogueStateTracker.from_events("test", evts=[SlotSet("foo", "bar")])
    command = SetSlotCommand(name="foo", value="bar")

    assert command.run_command_on_tracker(tracker, FlowsList(flows=[]), tracker) == []


def test_run_command_sets_slot_if_asked_for():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
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
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_foo",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    command = SetSlotCommand(name="foo", value="foofoo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("foo", "foofoo")]


def test_run_command_skips_set_slot_if_slot_was_not_asked_for():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
                steps:
                - id: collect_foo
                  collect: foo
                  next: collect_bar
                - id: collect_bar
                  ask_before_filling: true
                  collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_foo",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    command = SetSlotCommand(name="bar", value="barbar")

    # can't be set, because the collect information step requires the slot
    # to be asked before it can be filled
    assert command.run_command_on_tracker(tracker, all_flows, tracker) == []


def test_run_command_can_set_slots_before_asking():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
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
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_foo",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    command = SetSlotCommand(name="bar", value="barbar")

    # CAN be set, because the collect information step does not require the slot
    # to be asked before it can be filled
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("bar", "barbar")]


def test_run_command_can_set_slot_that_was_already_asked_in_the_past():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
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
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_bar",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    # set the slot for a collect information that was asked in the past
    # this isn't how we'd usually use this command as this should be converted
    # to a "correction" to trigger a correction pattern rather than directly
    # setting the slot.
    command = SetSlotCommand(name="foo", value="foofoo")
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("foo", "foofoo")]


def test_run_command_skips_setting_unknown_slot():
    all_flows = flows_from_str(
        """
        flows:
            my_flow:
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
        evts=[
            SlotSet(
                DIALOGUE_STACK_SLOT,
                [
                    {
                        "type": "flow",
                        "flow_id": "my_flow",
                        "step_id": "collect_bar",
                        "frame_id": "some-frame-id",
                    },
                ],
            ),
        ],
    )
    # set the slot for a collect information that was asked in the past
    command = SetSlotCommand(name="unknown", value="unknown")

    assert command.run_command_on_tracker(tracker, all_flows, tracker) == []
