import dataclasses

from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    UserFlowStackFrame,
)


def test_dialogue_stack_from_dict():
    stack = DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": "foo",
                "step_id": "first_step",
                "frame_id": "some-frame-id",
            },
            {
                "type": "pattern_collect_information",
                "collect": "foo",
                "frame_id": "some-other-id",
                "step_id": "START",
                "flow_id": "pattern_collect_information",
                "utter": "utter_ask_foo",
                "collect_action": "action_ask_foo",
            },
        ]
    )

    assert len(stack.frames) == 2

    assert stack.frames[0] == UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    assert stack.frames[1] == CollectInformationPatternFlowStackFrame(
        collect="foo",
        frame_id="some-other-id",
        utter="utter_ask_foo",
        collect_action="action_ask_foo",
    )


def test_dialogue_stack_move_frames_to_top():
    """Test moving frames to the top of the stack."""
    stack = DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": "foo",
                "step_id": "first_step",
                "frame_id": "frame-1",
            },
            {
                "type": "flow",
                "flow_id": "bar",
                "step_id": "second_step",
                "frame_id": "frame-2",
            },
            {
                "type": "flow",
                "flow_id": "baz",
                "step_id": "third_step",
                "frame_id": "frame-3",
            },
        ]
    )

    # Move the first frame to the top
    frame_to_move = stack.frames[0]
    stack.move_frames_to_top([frame_to_move])

    # Check that the moved frame is now at the top
    assert stack.frames[-1] == frame_to_move
    assert len(stack.frames) == 3  # All frames should still be present

    # Move multiple frames to the top
    frames_to_move = [stack.frames[0], stack.frames[1]]
    stack.move_frames_to_top(frames_to_move)

    # Check that the moved frames are at the top in the correct order
    assert stack.frames[-2:] == frames_to_move


def test_dialogue_stack_move_frames_to_top_preserves_order():
    """Test that move_frames_to_top preserves the relative order of moved frames."""
    stack = DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": "foo",
                "step_id": "first_step",
                "frame_id": "frame-1",
            },
            {
                "type": "flow",
                "flow_id": "bar",
                "step_id": "second_step",
                "frame_id": "frame-2",
            },
            {
                "type": "flow",
                "flow_id": "baz",
                "step_id": "third_step",
                "frame_id": "frame-3",
            },
        ]
    )

    # Move frames in reverse order to test order preservation
    frames_to_move = [stack.frames[2], stack.frames[1]]  # baz, bar
    stack.move_frames_to_top(frames_to_move)

    # Check that the moved frames are at the top in the correct order
    assert stack.frames[-2].flow_id == "baz"
    assert stack.frames[-1].flow_id == "bar"


def test_dialogue_stack_move_frames_to_top_with_pattern_frames():
    """Test moving frames to the top when pattern frames are present."""
    stack = DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": "foo",
                "step_id": "first_step",
                "frame_id": "frame-1",
            },
            {
                "type": "pattern_collect_information",
                "flow_id": "pattern_collect_information",
                "step_id": "START",
                "frame_id": "pattern-frame",
                "collect": "first_step",
                "collect_action": "action_ask_first_step",
                "utter": "utter_ask_first_step",
            },
            {
                "type": "flow",
                "flow_id": "bar",
                "step_id": "second_step",
                "frame_id": "frame-2",
            },
        ]
    )

    # Move foo and pattern to the top
    frames_to_move = [stack.frames[0], stack.frames[1]]  # foo, pattern
    stack.move_frames_to_top(frames_to_move)

    # Check that foo and pattern are at the top
    assert stack.frames[-2].flow_id == "foo"
    assert stack.frames[-1].flow_id == "pattern_collect_information"
    assert stack.frames[0].flow_id == "bar"  # bar should be at bottom


def test_dialogue_stack_from_dict_handles_empty():
    stack = DialogueStack.from_dict([])
    assert stack.frames == []


def test_dialogue_stack_as_dict():
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            CollectInformationPatternFlowStackFrame(
                collect="foo",
                frame_id="some-other-id",
                utter="utter_ask_foo",
                collect_action="action_ask_foo",
            ),
        ]
    )

    assert stack.as_dict() == [
        {
            "type": "flow",
            "flow_id": "foo",
            "frame_type": "regular",
            "step_id": "first_step",
            "frame_id": "some-frame-id",
        },
        {
            "type": "pattern_collect_information",
            "collect": "foo",
            "frame_id": "some-other-id",
            "step_id": "START",
            "flow_id": "pattern_collect_information",
            "rejections": None,
            "utter": "utter_ask_foo",
            "collect_action": "action_ask_foo",
        },
    ]


def test_dialogue_stack_as_dict_handles_empty():
    stack = DialogueStack.empty()
    assert stack.as_dict() == []


def test_push_to_empty_stack():
    stack = DialogueStack.empty()
    stack.push(
        UserFlowStackFrame(
            flow_id="foo", step_id="first_step", frame_id="some-frame-id"
        )
    )

    assert stack.frames == [
        UserFlowStackFrame(
            flow_id="foo", step_id="first_step", frame_id="some-frame-id"
        )
    ]


def test_push_to_non_empty_stack():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )

    stack = DialogueStack(frames=[user_frame])
    stack.push(pattern_frame)
    assert stack.top() == pattern_frame
    assert stack.frames == [user_frame, pattern_frame]


def test_push_to_index():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )

    stack = DialogueStack(frames=[user_frame])
    stack.push(pattern_frame, index=0)
    assert stack.top() == user_frame
    assert stack.frames == [pattern_frame, user_frame]


def test_dialogue_stack_update():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    updated_user_frame = dataclasses.replace(user_frame, step_id="second_step")
    stack.update(updated_user_frame)
    assert stack.top() == updated_user_frame
    assert stack.frames == [updated_user_frame]


def test_update_empty_stack():
    stack = DialogueStack.empty()
    stack.update(
        UserFlowStackFrame(
            flow_id="foo", step_id="first_step", frame_id="some-frame-id"
        )
    )

    assert stack.frames == [
        UserFlowStackFrame(
            flow_id="foo", step_id="first_step", frame_id="some-frame-id"
        )
    ]


def test_pop_frame():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )

    stack = DialogueStack.empty()
    stack.push(user_frame)
    stack.push(pattern_frame)
    assert stack.pop() == pattern_frame
    assert stack.frames == [user_frame]


def test_top_empty_stack():
    stack = DialogueStack.empty()
    assert stack.top() is None


def test_top():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )

    stack = DialogueStack.empty()
    stack.push(user_frame)
    stack.push(pattern_frame)
    assert stack.top() == pattern_frame


def test_get_current_context_empty_stack():
    stack = DialogueStack.empty()
    assert stack.current_context() == {}


def test_get_current_context():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo",
        frame_id="some-other-id",
        utter="utter_ask_foo",
        collect_action="action_ask_foo",
    )

    stack = DialogueStack.empty()
    stack.push(user_frame)
    stack.push(pattern_frame)
    assert stack.current_context() == {
        "flow_id": "foo",
        "frame_id": "some-frame-id",
        "frame_type": "regular",
        "step_id": "first_step",
        "type": "flow",
        "collect": "foo",
        "utter": "utter_ask_foo",
        "collect_action": "action_ask_foo",
        "rejections": None,
    }


def test_is_empty_on_empty():
    stack = DialogueStack.empty()
    assert stack.is_empty() is True


def test_is_empty_on_non_empty():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    assert stack.is_empty() is False


def test_create_stack_patch_with_empty():
    empty_stack = DialogueStack.empty()
    assert empty_stack.create_stack_patch(empty_stack) is None


def test_create_stack_patch_with_same_stack_is_none():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    assert stack.create_stack_patch(stack) is None


def test_create_stack_patch_with_different_stack():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])

    updated_frame = dataclasses.replace(user_frame, step_id="second_step")
    updated_stack = DialogueStack(frames=[updated_frame])

    patch = stack.create_stack_patch(updated_stack)
    assert patch == '[{"op": "replace", "path": "/0/step_id", "value": "second_step"}]'


def test_create_stack_patch_with_different_stack_starting_empty():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack.empty()

    updated_stack = DialogueStack(frames=[user_frame])

    patch = stack.create_stack_patch(updated_stack)
    expected_patch = (
        '[{"op": "add", "path": "/0", "value": {"frame_id": "some-frame-id", '
        '"flow_id": "foo", "step_id": "first_step", "frame_type": "regular", '
        '"type": "flow"}}]'
    )
    assert patch == expected_patch


def test_stack_update_from_patch_starting_empty():
    stack = DialogueStack.empty()

    patch = (
        '[{"op": "add", "path": "/0", "value": {'
        '"frame_id": "some-frame-id", "flow_id": "foo", '
        '"step_id": "first_step", "frame_type": "regular", "type": "flow"}}]'
    )

    updated_stack = stack.update_from_patch(patch)

    assert updated_stack.frames == [
        UserFlowStackFrame(
            flow_id="foo", step_id="first_step", frame_id="some-frame-id"
        )
    ]


def test_stack_update_from_existing_stack():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])

    patch = '[{"op": "replace", "path": "/0/step_id", "value": "second_step"}]'
    updated_stack = stack.update_from_patch(patch)

    assert updated_stack.frames == [
        UserFlowStackFrame(
            flow_id="foo", step_id="second_step", frame_id="some-frame-id"
        )
    ]
