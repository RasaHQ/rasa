import pytest

from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    end_top_user_flow,
    filled_slots_for_active_flow,
    get_collect_steps_excluding_ask_before_filling_for_active_flow,
    top_flow_frame,
    top_user_flow_frame,
    user_flows_on_the_stack,
)
from rasa.shared.core.events import SlotSet, UserUttered, FlowStarted
from rasa.shared.core.flows.yaml_flows_io import flows_from_str
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import update_tracker_with_path_through_flow


def test_top_flow_frame_ignores_pattern():
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(
        frames=[
            user_frame,
            pattern_frame,
        ]
    )

    assert top_flow_frame(stack, ignore_collect_information_pattern=True) == user_frame


def test_top_flow_frame_ignores_call():
    call_frame = UserFlowStackFrame(
        flow_id="bar",
        step_id="first_step",
        frame_id="some-call-id",
        frame_type=FlowStackFrameType.CALL,
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(
        frames=[
            user_frame,
            call_frame,
        ]
    )

    assert top_flow_frame(stack) == user_frame


def test_top_flow_frame_uses_pattern():
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame, pattern_frame])

    assert (
        top_flow_frame(stack, ignore_collect_information_pattern=False) == pattern_frame
    )


def test_top_flow_frame_handles_empty():
    stack = DialogueStack.empty()
    assert top_flow_frame(stack) is None


def test_top_user_flow_frame():
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame, pattern_frame])

    assert top_user_flow_frame(stack) == user_frame


def test_top_user_flow_frame_handles_empty():
    stack = DialogueStack.empty()
    assert top_user_flow_frame(stack) is None


def test_user_flows_on_the_stack():
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    another_user_frame = UserFlowStackFrame(
        flow_id="bar", step_id="first_step", frame_id="some-other-other-id"
    )
    stack = DialogueStack(frames=[user_frame, pattern_frame, another_user_frame])

    assert user_flows_on_the_stack(stack) == {"foo", "bar"}


def test_user_flows_on_the_stack_handles_empty():
    stack = DialogueStack.empty()
    assert user_flows_on_the_stack(stack) == set()


def test_filled_slots_for_active_flow_start():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - id: collect_foo
              collect: foo
            - id: collect_bar
              collect: bar
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events("test", [])
    update_tracker_with_path_through_flow(
        tracker, flow_id="my_flow", step_ids=["START"]
    )

    assert filled_slots_for_active_flow(tracker, all_flows) == set()


def test_filled_slots_for_active_flow_end():
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
              next: collect_baz
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            SlotSet("foo", "foofoo"),
            SlotSet("bar", "barbar"),
        ],
    )
    update_tracker_with_path_through_flow(
        tracker,
        flow_id="my_flow",
        step_ids=["collect_foo", "collect_bar", "collect_baz"],
    )
    assert filled_slots_for_active_flow(tracker, all_flows) == {"foo", "bar", "baz"}


def test_filled_slots_for_active_flow_handles_empty():
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
              next: collect_baz
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events("test", [])
    assert filled_slots_for_active_flow(tracker, all_flows) == set()


# TODO: ENG-687 fix this test by adding an abstraction for creating proper trackers
@pytest.mark.skip
def test_filled_slots_for_active_flow_skips_user_uttered():
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
              next: collect_baz
            - id: collect_baz
              collect: baz
        """
    )

    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[
            FlowStarted("my_flow"),
            SlotSet("foo", "foofoo"),
            SlotSet("not_valid_slot", "some_value"),
            UserUttered("some message"),
        ],
    )
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "flow_id": "my_flow",
                    "step_id": "collect_bar",
                    "frame_id": "some-frame-id",
                }
            ]
        )
    )

    assert filled_slots_for_active_flow(tracker, all_flows) == {"foo"}


# TODO: ENG-687 fix this test by adding an abstraction for creating proper trackers
@pytest.mark.skip
def test_filled_slots_for_active_flow_only_collects_till_top_most_user_flow_frame():
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
              next: collect_baz
            - id: collect_baz
              collect: baz
          my_other_flow:
            name: foo flow
            steps:
            - id: collect_foo2
              collect: foo2
              next: collect_bar2
            - id: collect_bar2
              collect: bar2
              next: collect_baz2
            - id: collect_baz2
              collect: baz2
        """
    )

    tracker = DialogueStateTracker.from_events("test", evts=[])
    update_tracker_with_path_through_flow(
        tracker,
        flow_id="my_flow",
        step_ids=["collect_foo", "collect_bar", "collect_baz"],
        frame_id="some-frame-id",
    )
    update_tracker_with_path_through_flow(
        tracker,
        flow_id="my_other_flow",
        step_ids=["collect_foo2", "collect_bar2"],
        frame_id="some-other-id",
    )

    assert filled_slots_for_active_flow(tracker, all_flows) == {"foo2", "bar2"}


def test_end_top_user_flow():
    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    stack = DialogueStack(frames=[user_frame, pattern_frame])

    updated_stack = end_top_user_flow(stack)

    assert len(updated_stack.frames) == 2

    assert updated_stack.frames[0] == UserFlowStackFrame(
        flow_id="my_flow", step_id="NEXT:END", frame_id="some-frame-id"
    )
    assert updated_stack.frames[1] == CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id", step_id="NEXT:END"
    )


def test_end_top_user_flow_only_ends_topmost_user_frame():
    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    other_user_frame = UserFlowStackFrame(
        flow_id="my_other_flow", step_id="collect_bar2", frame_id="some-other-id"
    )
    stack = DialogueStack(frames=[other_user_frame, user_frame])

    updated_stack = end_top_user_flow(stack)

    assert len(updated_stack.frames) == 2

    assert updated_stack.frames[0] == UserFlowStackFrame(
        flow_id="my_other_flow", step_id="collect_bar2", frame_id="some-other-id"
    )
    assert updated_stack.frames[1] == UserFlowStackFrame(
        flow_id="my_flow", step_id="NEXT:END", frame_id="some-frame-id"
    )


def test_end_top_user_flow_handles_empty():
    stack = DialogueStack.empty()
    end_top_user_flow(stack)

    assert len(stack.frames) == 0


def test_get_collect_steps_excluding_ask_before_filling_for_active_flow() -> None:
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - collect: foo
            - collect: bar
              ask_before_filling: true
            - collect: baz
          other_flow:
            description: test other flow
            name: abc flow
            steps:
            - collect: abc
            - collect: xyz
              ask_before_filling: true
            - collect: fgh
        """
    )
    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    slots = get_collect_steps_excluding_ask_before_filling_for_active_flow(
        stack, all_flows
    )
    assert slots == {"foo", "baz"}


def test_get_collect_steps_excluding_ask_before_filling_empty_stack() -> None:
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
            description: test my flow
            name: foo flow
            steps:
            - collect: foo
            - collect: bar
              ask_before_filling: true
            - collect: baz
          other_flow:
            description: test other flow
            name: abc flow
            steps:
            - collect: abc
            - collect: xyz
              ask_before_filling: true
            - collect: fgh
        """
    )
    stack = DialogueStack.empty()
    slots = get_collect_steps_excluding_ask_before_filling_for_active_flow(
        stack, all_flows
    )
    assert slots == set()
