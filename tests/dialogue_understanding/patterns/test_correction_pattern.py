import dataclasses
from pytest import CaptureFixture

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    ActionCorrectFlowSlot,
    CorrectionPatternFlowStackFrame,
    create_termination_frames_for_missing_frames,
    find_previous_state_to_reset_to,
    reset_stack_on_tracker_to_prior_state,
    set_topmost_flow_frame_to_continue,
    slice_of_stack_below_target,
    slice_of_stack_with_target_and_above,
)
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker
from tests.dialogue_understanding.conftest import (
    advance_top_tracker_flow,
    update_tracker_with_path_through_flow,
)


async def test_correction_pattern_flow_stack_frame_type() -> None:
    frame = CorrectionPatternFlowStackFrame()
    assert frame.type() == "pattern_correction"


async def test_correction_pattern_flow_stack_frame_from_dict() -> None:
    frame = CorrectionPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "corrected_slots": {"foo": "bar"},
            "is_reset_only": False,
            "reset_flow_id": None,
            "reset_step_id": None,
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.corrected_slots == {"foo": "bar"}
    assert frame.is_reset_only is False
    assert frame.reset_flow_id is None
    assert frame.reset_step_id is None
    assert frame.flow_id == "pattern_correction"
    assert frame.type() == "pattern_correction"


async def test_action_correct_flow_slot_no_active_flow(capsys: CaptureFixture) -> None:
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        Domain.empty(),
    )
    assert events == []
    assert "action.correct_flow_slot.no_active_flow" in capsys.readouterr().out


async def test_action_correct_flow_slot_no_correct_frame(
    capsys: CaptureFixture,
) -> None:
    domain = Domain.empty()
    stack = DialogueStack(
        frames=[UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")]
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(stack)
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert events == []
    assert "action.correct_flow_slot.no_correction_frame" in capsys.readouterr().out


async def test_action_correct_flow_slot_no_reset_step_id() -> None:
    domain = Domain.empty()
    user_start_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="START", frame_id="target_frame"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="some_step_id", frame_id="target_frame"
    )
    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="foo_flow",
        reset_step_id=None,
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    previous_stack = DialogueStack(frames=[user_start_frame])
    tracker.update_stack(previous_stack)
    stack = DialogueStack(frames=[user_frame, collect_info_frame, correction_frame])
    tracker.update_stack(stack)
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 3

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "START"

    frame = updated_stack.frames[1]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot() -> None:
    domain = Domain.empty()
    user_start_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="ask_some_slot", frame_id="target_frame"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="some_step_id", frame_id="target_frame"
    )
    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="foo_flow",
        reset_step_id="ask_some_slot",
    )
    stack = DialogueStack(frames=[user_frame, collect_info_frame, correction_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(DialogueStack(frames=[user_start_frame]))
    tracker.update_stack(stack)
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 3

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "NEXT:ask_some_slot"

    frame = updated_stack.frames[1]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot_with_call() -> None:
    domain = Domain.empty()
    user_start_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="ask_some_slot", frame_id="target_frame"
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="some_step_id", frame_id="target_frame"
    )
    call_frame = UserFlowStackFrame(
        flow_id="called_flow",
        step_id="some_step_id",
        frame_type=FlowStackFrameType.CALL,
    )
    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="foo_flow",
        reset_step_id="ask_some_slot",
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )

    tracker.update_stack(DialogueStack(frames=[user_start_frame]))

    tracker.update_stack(
        DialogueStack(
            frames=[user_frame, call_frame, collect_info_frame, correction_frame]
        )
    )

    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 4

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "NEXT:ask_some_slot"

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "called_flow"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[3]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot_with_call_within_the_call() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="some_step_id")

    call_start_frame = UserFlowStackFrame(
        flow_id="called_flow",
        step_id="ask_some_slot",
        frame_type=FlowStackFrameType.CALL,
        frame_id="call_frame",
    )
    call_frame = UserFlowStackFrame(
        flow_id="called_flow",
        step_id="some_step_id",
        frame_type=FlowStackFrameType.CALL,
        frame_id="call_frame",
    )

    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="called_flow",
        reset_step_id="ask_some_slot",
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )

    tracker.update_stack(DialogueStack(frames=[user_frame, call_start_frame]))

    tracker.update_stack(
        DialogueStack(
            frames=[user_frame, call_frame, collect_info_frame, correction_frame]
        )
    )

    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 4

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "some_step_id"

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "called_flow"
    assert frame.step_id == "NEXT:ask_some_slot"
    assert frame.frame_id == "call_frame"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[3]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot_with_call_from_another_call() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="some_step_id")

    call_start_frame = UserFlowStackFrame(
        flow_id="called_flow",
        step_id="ask_some_slot",
        frame_type=FlowStackFrameType.CALL,
        frame_id="call_frame",
    )

    another_call_frame = UserFlowStackFrame(
        flow_id="another_called_flow",
        step_id="another_step_id",
        frame_type=FlowStackFrameType.CALL,
    )

    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="called_flow",
        reset_step_id="ask_some_slot",
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )

    tracker.update_stack(DialogueStack(frames=[user_frame, call_start_frame]))

    tracker.update_stack(
        DialogueStack(
            frames=[
                user_frame,
                another_call_frame,
                collect_info_frame,
                correction_frame,
            ]
        )
    )

    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 5

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "some_step_id"

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "called_flow"
    assert frame.step_id == "NEXT:ask_some_slot"
    assert frame.frame_id == "call_frame"

    frame = updated_stack.frames[2]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "another_called_flow"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[3]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[4]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot_during_interruption() -> None:
    domain = Domain.empty()

    base_frame = UserFlowStackFrame(flow_id="bar_flow", step_id="bar_some_step_id")

    user_start_frame = UserFlowStackFrame(
        flow_id="foo_flow",
        step_id="ask_some_slot",
        frame_id="target_frame",
        frame_type=FlowStackFrameType.INTERRUPT,
    )
    user_frame = UserFlowStackFrame(
        flow_id="foo_flow",
        step_id="some_step_id",
        frame_id="target_frame",
        frame_type=FlowStackFrameType.INTERRUPT,
    )
    collect_info_frame = CollectInformationPatternFlowStackFrame(
        flow_id="collect_info_pattern",
        step_id="1",
        collect="test_slot_confirm",
        utter="test_ask_slot_confrim",
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        corrected_slots={"foo": "bar"},
        is_reset_only=False,
        reset_flow_id="foo_flow",
        reset_step_id="ask_some_slot",
    )
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(DialogueStack(frames=[base_frame, user_start_frame]))
    stack = DialogueStack(
        frames=[base_frame, user_frame, collect_info_frame, correction_frame]
    )
    tracker.update_stack(stack)
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )

    assert len(events) == 2
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 4

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "bar_flow"
    assert frame.step_id == "bar_some_step_id"

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "NEXT:ask_some_slot"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CollectInformationPatternFlowStackFrame)
    assert frame.flow_id == "pattern_collect_information"
    assert frame.step_id == "NEXT:END"

    frame = updated_stack.frames[3]
    assert isinstance(frame, CorrectionPatternFlowStackFrame)
    assert frame.flow_id == "pattern_correction"
    assert frame.step_id == "1"
    assert frame.corrected_slots == {"foo": "bar"}

    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


def test_find_previous_state_to_reset_to_empty_tracker():
    tracker = DialogueStateTracker.from_events("test", [])
    assert find_previous_state_to_reset_to("foo_flow", "bar_step", tracker) is None


def test_find_previous_state_to_reset_to():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    update_tracker_with_path_through_flow(
        tracker,
        "foo_flow",
        ["collect_foo", "collect_bar", "collect_baz"],
        frame_id="some-frame-id",
    )

    previous_stack_state = find_previous_state_to_reset_to(
        "foo_flow", "collect_foo", tracker
    )

    assert previous_stack_state == DialogueStack.from_dict(
        [
            {
                "type": "flow",
                "flow_id": "foo_flow",
                "step_id": "collect_foo",
                "frame_id": "some-frame-id",
            }
        ]
    )


def test_slice_of_stack_below_target_empty_stack():
    target_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="collect_foo", frame_id="target_frame"
    )
    slice = slice_of_stack_below_target(target_frame, DialogueStack.empty())
    assert slice == []


def test_slice_of_stack_below_target():
    target_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="collect_foo", frame_id="target_frame"
    )
    stack_frames = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
            ),
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_bar", frame_id="some-other-id-2"
            ),
            target_frame,
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_baz", frame_id="some-other-id-3"
            ),
        ]
    )
    slice = slice_of_stack_below_target(target_frame, stack_frames)
    assert slice == stack_frames.frames[:2]


def test_slice_of_stack_with_target_and_above_empty_stack():
    target_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="collect_foo", frame_id="target_frame"
    )
    slice = slice_of_stack_with_target_and_above(target_frame, DialogueStack.empty())
    assert slice == []


def test_slice_of_stack_with_target_and_above():
    target_frame = UserFlowStackFrame(
        flow_id="foo_flow", step_id="collect_foo", frame_id="target_frame"
    )
    stack_frames = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
            ),
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_bar", frame_id="some-other-id-2"
            ),
            target_frame,
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_baz", frame_id="some-other-id-3"
            ),
        ]
    )
    slice = slice_of_stack_with_target_and_above(target_frame, stack_frames)
    assert slice == stack_frames.frames[2:]


def test_set_topmost_flow_frame_to_continue_empty():
    assert set_topmost_flow_frame_to_continue(stack_frames=[]) is None


def test_set_topmost_flow_frame_to_continue():
    stack_frames = [
        UserFlowStackFrame(
            flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
        ),
        UserFlowStackFrame(
            flow_id="foo_flow", step_id="collect_bar", frame_id="some-other-id-2"
        ),
        UserFlowStackFrame(
            flow_id="foo_flow", step_id="collect_baz", frame_id="some-other-id-3"
        ),
    ]

    original_stack_frames = [dataclasses.replace(frame) for frame in stack_frames]
    set_topmost_flow_frame_to_continue(stack_frames)

    # underlying frames should not change
    assert stack_frames[:2] == original_stack_frames[:2]

    # topmost frame should now be pointing to the a continue step for the original step
    assert stack_frames[2].step_id == "NEXT:collect_baz"
    assert stack_frames[2].frame_id == original_stack_frames[2].frame_id


def test_create_termination_frames_for_missing_frames_empty_frames_and_stack():
    assert (
        create_termination_frames_for_missing_frames(
            new_stack_frames=[], previous_stack=DialogueStack.empty()
        )
        == []
    )


def test_create_termination_frames_for_missing_frames_empty_previous_stack():
    new_stack_frames = [
        UserFlowStackFrame(
            flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
        ),
    ]

    termination_frames = create_termination_frames_for_missing_frames(
        new_stack_frames=new_stack_frames, previous_stack=DialogueStack.empty()
    )

    assert termination_frames == []


def test_create_termination_frames_for_missing_frames_empty_frames():
    previous_stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo_flow",
                step_id="collect_foo",
                frame_id="some-other-id-1",
                frame_type=FlowStackFrameType.CALL,
            ),
            UserFlowStackFrame(
                flow_id="bar_flow",
                step_id="collect_bar",
                frame_id="some-other-id-2",
                frame_type=FlowStackFrameType.CALL,
            ),
            UserFlowStackFrame(
                flow_id="baz_flow",
                step_id="collect_baz",
                frame_id="some-other-id-3",
                frame_type=FlowStackFrameType.CALL,
            ),
        ]
    )

    termination_frames = create_termination_frames_for_missing_frames(
        new_stack_frames=[], previous_stack=previous_stack
    )

    assert termination_frames == [
        UserFlowStackFrame(
            flow_id="foo_flow",
            step_id="NEXT:END",
            frame_id="some-other-id-1",
            frame_type=FlowStackFrameType.CALL,
        ),
        UserFlowStackFrame(
            flow_id="bar_flow",
            step_id="NEXT:END",
            frame_id="some-other-id-2",
            frame_type=FlowStackFrameType.CALL,
        ),
        UserFlowStackFrame(
            flow_id="baz_flow",
            step_id="NEXT:END",
            frame_id="some-other-id-3",
            frame_type=FlowStackFrameType.CALL,
        ),
    ]


def test_create_termination_frames_for_missing_collect_frames():
    previous_stack = DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
            ),
            CollectInformationPatternFlowStackFrame(
                step_id="1", collect="foo", frame_id="some-other-id-2"
            ),
            UserFlowStackFrame(
                flow_id="baz_flow", step_id="collect_baz", frame_id="some-other-id-3"
            ),
        ]
    )

    new_stack_frames = [
        UserFlowStackFrame(
            flow_id="foo_flow", step_id="collect_foo", frame_id="some-other-id-1"
        ),
        # collect information is removed
        UserFlowStackFrame(
            flow_id="baz_flow", step_id="collect_baz", frame_id="some-other-id-3"
        ),
        # other flow is newly added
        UserFlowStackFrame(
            flow_id="other_flow", step_id="collect_other", frame_id="some-other-id-4"
        ),
    ]

    termination_frames = create_termination_frames_for_missing_frames(
        new_stack_frames=new_stack_frames, previous_stack=previous_stack
    )

    assert termination_frames == [
        CollectInformationPatternFlowStackFrame(
            step_id="NEXT:END", collect="foo", frame_id="some-other-id-2"
        ),
    ]


def test_reset_stack_on_tracker_to_prior_state_no_frame_found():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    reset_flow_id = "foo"
    reset_step_id = "bar"

    assert (
        reset_stack_on_tracker_to_prior_state(reset_flow_id, reset_step_id, tracker)
        == DialogueStack.empty()
    )


def test_reset_stack_on_tracker_to_prior_state():
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    update_tracker_with_path_through_flow(
        tracker,
        "foo",
        ["collect_foo", "collect_bar", "collect_baz"],
        frame_id="some-frame-id",
    )

    reset_flow_id = "foo"
    reset_step_id = "collect_bar"

    stack = reset_stack_on_tracker_to_prior_state(reset_flow_id, reset_step_id, tracker)

    assert stack == DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="NEXT:collect_bar", frame_id="some-frame-id"
            ),
        ]
    )


def test_reset_stack_on_tracker_complex_with_calls():
    # this is an example where there are three flows:
    # ```
    # flows:
    #   foo:
    #     steps:
    #     - id: collect_foo
    #       collect: foo
    #     - id: first-call
    #       call: first-call
    #     - id: second-call
    #       call: second-call
    #   first-call:
    #     steps:
    #     - id: collect_call
    #       collect: call
    #   second-call:
    #     steps:
    #     - id: collect_second_call
    #       collect: second_call
    # ```
    # and we want to reset to the first call (step id `collect_call`)
    # while we are currently in the second call (step id `collect_second_call`).
    tracker = DialogueStateTracker.from_events(
        "test",
        evts=[],
    )
    update_tracker_with_path_through_flow(
        tracker, "foo", ["collect_foo", "first-call"], frame_id="some-frame-id"
    )

    # andvance the tracker to go into the first call
    stack = tracker.stack
    stack.push(
        UserFlowStackFrame(
            flow_id="first-call",
            step_id="collect_call",
            frame_id="some-call-id",
            frame_type=FlowStackFrameType.CALL,
        )
    )
    tracker.update_stack(stack)

    # andvance the tracker to end the first called flow and remove it from the stack
    advance_top_tracker_flow(tracker, "END")
    stack = tracker.stack
    stack.pop()
    tracker.update_stack(stack)

    # advance the parent flow to the second call
    advance_top_tracker_flow(tracker, "second-call")

    # add the second call to the stack
    stack = tracker.stack
    stack.push(
        UserFlowStackFrame(
            flow_id="second-call",
            step_id="collect_second_call",
            frame_id="second-call-id",
            frame_type=FlowStackFrameType.CALL,
        )
    )
    tracker.update_stack(stack)

    reset_flow_id = "first-call"
    # we want to reset to the first call while we are currently in the second call
    reset_step_id = "collect_call"

    stack = reset_stack_on_tracker_to_prior_state(reset_flow_id, reset_step_id, tracker)

    assert stack == DialogueStack(
        frames=[
            UserFlowStackFrame(
                flow_id="foo", step_id="first-call", frame_id="some-frame-id"
            ),
            UserFlowStackFrame(
                flow_id="first-call",
                step_id="NEXT:collect_call",
                frame_id="some-call-id",
                frame_type=FlowStackFrameType.CALL,
            ),
            UserFlowStackFrame(
                flow_id="second-call",
                step_id="NEXT:END",
                frame_id="second-call-id",
                frame_type=FlowStackFrameType.CALL,
            ),
        ]
    )
