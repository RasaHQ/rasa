import dataclasses
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import UserFlowStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import SlotSet


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
            },
        ]
    )

    assert len(stack.frames) == 2

    assert stack.frames[0] == UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    assert stack.frames[1] == CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id", utter="utter_ask_foo"
    )


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
        },
    ]


def test_dialogue_stack_as_event():
    # check that the stack gets persisted as an event storing the dict
    stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first_step", frame_id="some-frame-id"
            ),
            CollectInformationPatternFlowStackFrame(
                collect="foo",
                frame_id="some-other-id",
                utter="utter_ask_foo",
            ),
        ]
    )

    assert stack.persist_as_event() == SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())


def test_dialogue_stack_as_dict_handles_empty():
    stack = DialogueStack(frames=[])
    assert stack.as_dict() == []


def test_push_to_empty_stack():
    stack = DialogueStack(frames=[])
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
    stack = DialogueStack(frames=[])
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

    stack = DialogueStack(frames=[])
    stack.push(user_frame)
    stack.push(pattern_frame)
    assert stack.pop() == pattern_frame
    assert stack.frames == [user_frame]


def test_top_empty_stack():
    stack = DialogueStack(frames=[])
    assert stack.top() is None


def test_top():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id"
    )

    stack = DialogueStack(frames=[])
    stack.push(user_frame)
    stack.push(pattern_frame)
    assert stack.top() == pattern_frame


def test_get_current_context_empty_stack():
    stack = DialogueStack(frames=[])
    assert stack.current_context() == {}


def test_get_current_context():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    pattern_frame = CollectInformationPatternFlowStackFrame(
        collect="foo", frame_id="some-other-id", utter="utter_ask_foo"
    )

    stack = DialogueStack(frames=[])
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
        "rejections": None,
    }


def test_is_empty_on_empty():
    stack = DialogueStack(frames=[])
    assert stack.is_empty() is True


def test_is_empty_on_non_empty():
    user_frame = UserFlowStackFrame(
        flow_id="foo", step_id="first_step", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    assert stack.is_empty() is False
