from typing import Any, List

import pytest
from rasa.dialogue_understanding.commands.set_slot_command import (
    SetSlotExtractor,
    get_flows_predicted_to_start_from_tracker,
    Command,
    SetSlotCommand,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet, UserUttered
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import TextSlot
from rasa.dialogue_understanding.commands.start_flow_command import StartFlowCommand
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.nlu.constants import COMMANDS


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

    assert (
        command.run_command_on_tracker(tracker, FlowsList(underlying_flows=[]), tracker)
        == []
    )


def test_run_command_sets_slot_if_asked_for():
    slots = [TextSlot(name="foo", mappings=[])]

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

    tracker = DialogueStateTracker.from_events("test", evts=[], slots=slots)
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_foo",
                    "frame_id": "some-frame-id",
                },
            ],
        )
    )
    command = SetSlotCommand(name="foo", value="foofoo")

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("foo", "foofoo")]


def test_run_command_skips_set_slot_if_slot_was_not_asked_for():
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
                  ask_before_filling: true
                  collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_foo",
                    "frame_id": "some-frame-id",
                },
            ],
        )
    )
    command = SetSlotCommand(name="bar", value="barbar")

    # can't be set, because the collect information step requires the slot
    # to be asked before it can be filled
    assert command.run_command_on_tracker(tracker, all_flows, tracker) == []


def test_run_command_can_set_slots_before_asking():
    slots = [TextSlot(name="bar", mappings=[])]

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

    tracker = DialogueStateTracker.from_events("test", evts=[], slots=slots)
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_foo",
                    "frame_id": "some-frame-id",
                },
            ],
        )
    )
    command = SetSlotCommand(name="bar", value="barbar")

    # CAN be set, because the collect information step does not require the slot
    # to be asked before it can be filled
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("bar", "barbar")]


def test_run_command_can_set_slot_that_was_already_asked_in_the_past():
    slots = [TextSlot(name="foo", mappings=[])]

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

    tracker = DialogueStateTracker.from_events("test", evts=[], slots=slots)
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                },
            ],
        )
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
                description: test my flow
                steps:
                - id: collect_foo
                  collect: foo
                  next: collect_bar
                - id: collect_bar
                  collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                },
            ],
        )
    )
    # set the slot for a collect information that was asked in the past
    command = SetSlotCommand(name="unknown", value="unknown")

    assert command.run_command_on_tracker(tracker, all_flows, tracker) == []


def test_run_command_set_slot_of_startable_flows() -> None:
    slots = [TextSlot(name="baz", mappings=[])]

    all_flows = flows_from_str(
        """
        flows:
            my_flow:
                description: test my flow
                steps:
                - collect: foo
                - collect: bar
            another_flow:
                description: test another flow
                steps:
                - collect: baz
                - collect: qux
        """
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered(
                "start foo",
                None,
                None,
                {
                    COMMANDS: [
                        StartFlowCommand("another_flow").as_dict(),
                        SetSlotCommand("baz", "bazbaz").as_dict(),
                    ]
                },
            ),
        ],
        slots=slots,
    )
    command = SetSlotCommand(name="baz", value="bazbaz")
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("baz", "bazbaz")]


def test_run_command_set_slot_of_startable_flows_and_skip_the_rest() -> None:
    slots = [
        TextSlot(name="boom", mappings=[]),
        TextSlot(name="qux", mappings=[]),
        TextSlot(name="foo", mappings=[]),
    ]

    all_flows = flows_from_str(
        """
        flows:
            my_flow:
                description: test my flow
                steps:
                - collect: foo
                - collect: bar
            another_flow:
                description: test another flow
                steps:
                - collect: baz
                - collect: qux
            third_flow:
                description: test third flow
                steps:
                - collect: boom
                - collect: xyz
        """
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered(
                "start foo",
                None,
                None,
                {
                    COMMANDS: [
                        StartFlowCommand("my_flow").as_dict(),
                        StartFlowCommand("third_flow").as_dict(),
                        SetSlotCommand("foo", "foofoo").as_dict(),
                        SetSlotCommand("qux", "quxqux").as_dict(),
                        SetSlotCommand("boom", "boomboom").as_dict(),
                    ]
                },
            ),
        ],
        slots=slots,
    )
    command = SetSlotCommand(name="foo", value="foofoo")
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("foo", "foofoo")]

    command = SetSlotCommand(name="qux", value="quxqux")
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == []

    command = SetSlotCommand(name="boom", value="boomboom")
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet("boom", "boomboom")]


@pytest.mark.parametrize(
    "commands, expected",
    [
        (
            [
                StartFlowCommand("my_flow").as_dict(),
                SetSlotCommand("foo", "foofoo").as_dict(),
            ],
            ["my_flow"],
        ),
        (
            [
                StartFlowCommand("my_flow").as_dict(),
                StartFlowCommand("test_flow").as_dict(),
            ],
            ["my_flow", "test_flow"],
        ),
        ([SetSlotCommand("foo", "foofoo").as_dict()], []),
    ],
)
def test_get_flows_predicted_to_start_from_tracker(
    commands: List[Command], expected: List[str]
) -> None:
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            UserUttered("start foo", None, None, {COMMANDS: commands}),
        ],
    )
    assert get_flows_predicted_to_start_from_tracker(tracker) == expected


@pytest.mark.parametrize(
    "command, expected_value",
    [
        (
            SetSlotCommand(
                "foo", "foofoo", SetSlotExtractor.COMMAND_PAYLOAD_READER.value
            ),
            "foofoo",
        ),
        (SetSlotCommand("bar", "1", SetSlotExtractor.COMMAND_PAYLOAD_READER.value), 1),
        (
            SetSlotCommand(
                "baz", "true", SetSlotExtractor.COMMAND_PAYLOAD_READER.value
            ),
            True,
        ),
        (
            SetSlotCommand("qux", "a", SetSlotExtractor.COMMAND_PAYLOAD_READER.value),
            "a",
        ),
    ],
)
def test_run_command_on_tracker_slot_value_is_coerced_to_right_type(
    command: SetSlotCommand, expected_value: Any
) -> None:
    domain = Domain.from_yaml("""
    slots:
        foo:
          type: text
        bar:
          type: float
        baz:
          type: bool
        qux:
          type: categorical
          values:
            - a
            - b
    """)

    all_flows = flows_from_str(
        f"""
        flows:
            my_flow:
                description: test my flow
                steps:
                - id: collect_{command.name}
                  collect: {command.name}
        """
    )

    tracker = DialogueStateTracker.from_events("test_id", evts=[], slots=domain.slots)

    # ask for the slot
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": f"collect_{command.name}",
                    "frame_id": "some-frame-id",
                },
            ],
        )
    )
    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert events == [SlotSet(command.name, expected_value)]
