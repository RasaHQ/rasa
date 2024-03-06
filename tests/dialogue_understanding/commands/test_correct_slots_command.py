from typing import Any, Dict, List
import pytest
from rasa.dialogue_understanding.commands.correct_slots_command import (
    CorrectSlotsCommand,
    CorrectedSlot,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.events import Event
from tests.dialogue_understanding.conftest import update_tracker_with_path_through_flow
from tests.utilities import flows_from_str_with_defaults
from rasa.shared.core.events import DialogueStackUpdated, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.flows.yaml_flows_io import (
    flows_from_str,
    flows_from_str_including_defaults,
)
import jsonpatch


def test_command_name():
    # names of commands should not change as they are part of persisted
    # trackers
    assert CorrectSlotsCommand.command() == "correct slot"


def test_from_dict():
    assert CorrectSlotsCommand.from_dict(
        {"corrected_slots": [{"name": "foo", "value": "bar"}]}
    ) == CorrectSlotsCommand(corrected_slots=[CorrectedSlot(name="foo", value="bar")])


def test_from_dict_fails_if_missing_name_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({"corrected_slots": [{"value": "bar"}]})


def test_from_dict_fails_if_missing_value_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({"corrected_slots": [{"name": "foo"}]})


def test_from_dict_fails_if_missing_corrected_slots_parameter():
    with pytest.raises(ValueError):
        CorrectSlotsCommand.from_dict({})


def test_run_command_on_tracker_without_flows():
    tracker = DialogueStateTracker.from_events("test", evts=[])
    command = CorrectSlotsCommand(corrected_slots=[])

    assert command.run_command_on_tracker(tracker, [], tracker) == []


@pytest.fixture
def collect_foo_tracker_update() -> Dict[str, Any]:
    return {
        "frame_id": "HF1UKRND",
        "flow_id": "pattern_collect_information",
        "step_id": "START",
        "collect": "foo",
        "utter": "utter_ask_foo",
        "rejections": [],
        "type": "pattern_collect_information",
    }


@pytest.fixture
def collect_bar_tracker_update() -> Dict[str, Any]:
    return {
        "frame_id": "HF1UKRND",
        "flow_id": "pattern_collect_information",
        "step_id": "START",
        "collect": "bar",
        "utter": "utter_ask_bar",
        "rejections": [],
        "type": "pattern_collect_information",
    }


def test_run_command_on_tracker_correcting_previous_flow():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="foo", value="not-foofoo")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a correction pattern should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 2

    assert dialogue_stack_dump[1]["type"] == "pattern_correction"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["corrected_slots"] == {"foo": "not-foofoo"}
    assert dialogue_stack_dump[1]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[1]["reset_step_id"] == "collect_foo"
    assert dialogue_stack_dump[1]["is_reset_only"] is False


def test_run_command_on_tracker_correcting_current_flow(
    collect_foo_tracker_update: Dict[str, Any],
    collect_bar_tracker_update: Dict[str, Any],
):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
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
                collect_foo_tracker_update,
                collect_bar_tracker_update,
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                },
            ]
        )
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="bar", value="barbar")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    # flow should still be on the stack and a correction flow should have been added
    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 4

    assert dialogue_stack_dump[3]["type"] == "pattern_correction"
    assert dialogue_stack_dump[3]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[3]["step_id"] == "START"
    assert dialogue_stack_dump[3]["corrected_slots"] == {"bar": "barbar"}
    assert dialogue_stack_dump[3]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[3]["reset_step_id"] == "collect_bar"
    assert dialogue_stack_dump[3]["is_reset_only"] is False


# Skipped in https://rasahq.atlassian.net/browse/ENG-687
# To be fixed in https://rasahq.atlassian.net/browse/ENG-690
@pytest.mark.skip(reason="ENG-687")
@pytest.mark.parametrize(
    "corrected_slots, events, dialogue_stack",
    [
        (
            [CorrectedSlot(name="foobar", value="foobarfoobar")],
            [
                SlotSet("foo", "foo"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_bar",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="bar", value="barbar")],
            [
                SlotSet("foo", "foo"),
                SlotSet("foobar", "foobar"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foobar",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="foo", value="foofoo")],
            [],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foo",
                "frame_id": "some-frame-id",
            },
        ),
        (
            [CorrectedSlot(name="bar", value="barbar")],
            [
                SlotSet("foo", "foofoo"),
                SlotSet("bar", "bar"),
                SlotSet("foobar", "foobar"),
            ],
            {
                "type": "flow",
                "frame_type": "regular",
                "flow_id": "my_flow",
                "step_id": "collect_foobar",
                "frame_id": "some-frame-id",
            },
        ),
    ],
)
def test_run_command_on_tracker_correcting_invalid_slot(
    corrected_slots: List[CorrectedSlot],
    events: List[Event],
    dialogue_stack: Dict[str, str],
):
    all_flows = flows_from_str_with_defaults(
        """
        flows:
          my_flow:
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next:
              - if: foo == "foo"
                then: collect_bar
              - else: collect_foobar
            - id: collect_bar
              collect: bar
              next: END
            - id: collect_foobar
              collect: foobar
              next: END
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=events,
    )
    tracker.update_stack(DialogueStack.from_dict([dialogue_stack]))
    command = CorrectSlotsCommand(corrected_slots=corrected_slots)

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 0


def test_run_command_on_tracker_correcting_slot_with_asked_before_filling():
    all_flows = flows_from_str_with_defaults(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              ask_before_filling: true
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="foo", value="foofoo")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.is_reset_only
    assert frame.corrected_slots == {"foo": "foofoo"}
    assert frame.step_id == "START"
    assert frame.reset_step_id == "collect_foo"


def test_run_command_on_tracker_correcting_during_a_correction():
    all_flows = flows_from_str_including_defaults(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )
    stack = tracker.stack
    stack.push(
        CorrectionPatternFlowStackFrame(
            corrected_slots={"foo": "not-foofoo"},
            step_id="0_action_correct_flow_slot",
            reset_flow_id="my_flow",
            reset_step_id="collect_foo",
            is_reset_only=False,
            frame_id="some-other-id",
        )
    )
    tracker.update_stack(stack)

    command = CorrectSlotsCommand(
        corrected_slots=[CorrectedSlot(name="bar", value="barbar")]
    )

    events = command.run_command_on_tracker(tracker, all_flows, tracker)
    assert len(events) == 1

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    patch = jsonpatch.JsonPatch.from_string(dialogue_stack_event.update)
    dialogue_stack_dump = patch.apply(tracker.stack.as_dict())

    assert isinstance(dialogue_stack_dump, list) and len(dialogue_stack_dump) == 3

    assert dialogue_stack_dump[1]["type"] == "pattern_correction"
    assert dialogue_stack_dump[1]["flow_id"] == "pattern_correction"
    assert dialogue_stack_dump[1]["step_id"] == "START"
    assert dialogue_stack_dump[1]["corrected_slots"] == {"bar": "barbar"}
    assert dialogue_stack_dump[1]["reset_flow_id"] == "my_flow"
    assert dialogue_stack_dump[1]["reset_step_id"] == "collect_bar"

    assert dialogue_stack_dump[2]["type"] == "pattern_correction"
    assert dialogue_stack_dump[2]["corrected_slots"] == {"foo": "not-foofoo"}


def test_determine_index_for_new_correction_frame_handles_empty_stack():
    stack = DialogueStack.empty()
    top_flow_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 0
    )


def test_index_for_correction_handles_non_correction_pattern_at_the_top_of_stack():
    top_flow_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 2
    )


def test_index_for_correction_handles_correction_pattern_at_the_top_of_stack():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    # new correction pattern should be inserted "under" the existing correction pattern
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 1
    )


def test_index_for_correction_handles_call_and_existing_correction_pattern():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            UserFlowStackFrame(
                flow_id="bar",
                step_id="first_step",
                frame_id="some-call-id",
                frame_type=FlowStackFrameType.CALL,
            ),
            top_flow_frame,
        ]
    )
    # new correction pattern should be inserted "under" the existing correction pattern
    assert (
        CorrectSlotsCommand.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        == 2
    )


def test_end_previous_correction():
    top_flow_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={"foo": "not-foofoo"},
    )
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            top_flow_frame,
        ]
    )
    CorrectSlotsCommand.end_previous_correction(top_flow_frame, stack)
    # the previous pattern should be about to end
    assert stack.frames[1].step_id == "NEXT:END"
    # make sure the user flow has not been modified
    assert stack.frames[0].step_id == "first_step"


def test_end_previous_correction_no_correction_present():
    top_flow_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[top_flow_frame])
    CorrectSlotsCommand.end_previous_correction(top_flow_frame, stack)
    # make sure the user flow has not been modified
    assert stack.frames[0].step_id == "first_step"


@pytest.mark.parametrize(
    "updated_slots, expected_step_id",
    [
        (
            ["foo", "bar"],
            "collect_foo",
        ),
        (
            ["bar", "foo"],
            "collect_foo",
        ),
        (
            ["bar"],
            "collect_bar",
        ),
        (
            ["foo"],
            "collect_foo",
        ),
        (
            [],
            None,
        ),
    ],
)
def test_find_earliest_updated_collect_info(
    updated_slots: List[str],
    expected_step_id: str,
):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            name: foo flow
            description: foo flow
            steps:
            - id: collect_foo
              collect: foo
              next: collect_bar
            - id: collect_bar
              collect: bar
              next: collect_baz
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events("bot", evts=[])
    update_tracker_with_path_through_flow(
        tracker, "my_flow", ["collect_foo", "collect_bar"]
    )

    step = CorrectSlotsCommand.find_earliest_updated_collect_info(
        updated_slots, all_flows, tracker
    )
    if expected_step_id is not None:
        assert step is not None
        assert step.step.id == expected_step_id
    else:
        assert step is None


@pytest.mark.parametrize(
    "proposed_slots, expected",
    [
        ({}, True),
        ({"foo": "foofoo"}, True),
        ({"bar": "barbar"}, False),
        ({"foo": "foofoo", "bar": "barbar"}, False),
    ],
)
def test_are_all_slots_reset_only(proposed_slots: Dict[str, Any], expected: bool):
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
              ask_before_filling: true
              next: collect_bar
            - id: collect_bar
              collect: bar
        """
    )
    assert (
        CorrectSlotsCommand.are_all_slots_reset_only(proposed_slots, all_flows)
        == expected
    )
