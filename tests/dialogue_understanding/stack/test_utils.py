from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.chit_chat_frame import ChitChatStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.dialogue_understanding.stack.utils import (
    end_top_user_flow,
    filled_slots_for_active_flow,
    top_flow_frame,
    top_user_flow_frame,
    user_flows_on_the_stack,
)
from rasa.shared.core.flows.yaml_flows_io import flows_from_str


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
    stack = DialogueStack(frames=[])
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
    stack = DialogueStack(frames=[])
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
    stack = DialogueStack(frames=[])
    assert user_flows_on_the_stack(stack) == set()


def test_filled_slots_for_active_flow():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
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

    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])

    assert filled_slots_for_active_flow(stack, all_flows) == {"foo", "bar"}


def test_filled_slots_for_active_flow_handles_empty():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
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

    stack = DialogueStack(frames=[])
    assert filled_slots_for_active_flow(stack, all_flows) == set()


def test_filled_slots_for_active_flow_skips_chitchat():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
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

    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    chitchat_frame = ChitChatStackFrame(frame_id="some-other-id")
    stack = DialogueStack(frames=[user_frame, chitchat_frame])

    assert filled_slots_for_active_flow(stack, all_flows) == {"foo", "bar"}


def test_filled_slots_for_active_flow_only_collects_till_top_most_user_flow_frame():
    all_flows = flows_from_str(
        """
        flows:
          my_flow:
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

    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    another_user_frame = UserFlowStackFrame(
        flow_id="my_other_flow", step_id="collect_bar2", frame_id="some-other-id"
    )
    stack = DialogueStack(frames=[another_user_frame, user_frame])

    assert filled_slots_for_active_flow(stack, all_flows) == {"foo", "bar"}


def test_end_top_user_flow():
    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )
    stack = DialogueStack(frames=[user_frame, pattern_frame])

    end_top_user_flow(stack)

    assert len(stack.frames) == 2

    assert stack.frames[0] == UserFlowStackFrame(
        flow_id="my_flow", step_id="NEXT:END", frame_id="some-frame-id"
    )
    assert stack.frames[1] == CollectInformationPatternFlowStackFrame(
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

    end_top_user_flow(stack)

    assert len(stack.frames) == 2

    assert stack.frames[0] == UserFlowStackFrame(
        flow_id="my_other_flow", step_id="collect_bar2", frame_id="some-other-id"
    )
    assert stack.frames[1] == UserFlowStackFrame(
        flow_id="my_flow", step_id="NEXT:END", frame_id="some-frame-id"
    )


def test_end_top_user_flow_handles_empty():
    stack = DialogueStack(frames=[])
    end_top_user_flow(stack)

    assert len(stack.frames) == 0
