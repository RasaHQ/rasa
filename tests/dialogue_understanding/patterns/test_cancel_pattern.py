from pytest import CaptureFixture

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.cancel import (
    ActionCancelFlow,
    CancelPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import DialogueStackUpdated
from rasa.shared.core.trackers import DialogueStateTracker


async def test_cancel_pattern_flow_stack_frame_type() -> None:
    frame = CancelPatternFlowStackFrame()
    assert frame.type() == "pattern_cancel_flow"


async def test_cancel_pattern_flow_stack_frame_from_dict() -> None:
    frame = CancelPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "canceled_name": "x_flow",
            "canceled_frames": ["x_frame"],
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.canceled_name == "x_flow"
    assert frame.canceled_frames == ["x_frame"]
    assert frame.flow_id == "pattern_cancel_flow"
    assert frame.type() == "pattern_cancel_flow"


async def test_action_cancel_flow_no_active_flow(capsys: CaptureFixture) -> None:
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionCancelFlow()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        Domain.empty(),
    )
    assert events == []
    assert "action.cancel_flow.no_active_flow" in capsys.readouterr().out


async def test_action_cancel_flow_no_cancel_frame(capsys: CaptureFixture) -> None:
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
    action = ActionCancelFlow()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert events == []
    assert "action.cancel_flow.no_cancel_frame" in capsys.readouterr().out


async def test_action_cancel_flow_frame_not_found(capsys: CaptureFixture) -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="some-id")
    cancel_frame = CancelPatternFlowStackFrame(
        step_id="1",
        frame_id="test_id",
        canceled_name="foo_flow",
        canceled_frames=["some-other-id"],
    )
    stack = DialogueStack(frames=[user_frame, cancel_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(stack)
    action = ActionCancelFlow()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert len(events) == 0
    assert "action.cancel_flow.frame_not_found" in capsys.readouterr().out


async def test_action_cancel_flow_single_cancelled_frame() -> None:
    domain = Domain.empty()
    user_frame1 = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-id"
    )
    user_frame2 = UserFlowStackFrame(
        flow_id="bar_flow", step_id="1", frame_id="some-other-id"
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        canceled_name="bar_flow",
        canceled_frames=["some-other-id"],
    )
    stack = DialogueStack(frames=[user_frame1, user_frame2, cancel_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(stack)
    action = ActionCancelFlow()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        Domain.empty(),
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 3

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "1"
    assert frame.frame_id == "some-id"

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "bar_flow"
    assert frame.step_id == "NEXT:END"
    assert frame.frame_id == "some-other-id"

    frame = updated_stack.frames[2]
    assert isinstance(frame, CancelPatternFlowStackFrame)
    assert frame.flow_id == "pattern_cancel_flow"
    assert frame.step_id == "1"
    assert frame.frame_id == "test_id"
    assert frame.canceled_name == "bar_flow"


async def test_action_cancel_flow_multiple_cancelled_frame() -> None:
    domain = Domain.empty()
    user_frame1 = UserFlowStackFrame(
        flow_id="foo_flow", step_id="1", frame_id="some-id"
    )
    user_frame2 = UserFlowStackFrame(
        flow_id="bar_flow", step_id="1", frame_id="some-other-id"
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="test_id",
        step_id="1",
        canceled_name="bar_flow",
        canceled_frames=["some-other-id", "some-id"],
    )
    stack = DialogueStack(frames=[user_frame1, user_frame2, cancel_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[],
    )
    tracker.update_stack(stack)
    action = ActionCancelFlow()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        Domain.empty(),
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 3

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo_flow"
    assert frame.step_id == "NEXT:END"
    assert frame.frame_id == "some-id"
    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "bar_flow"
    assert frame.step_id == "NEXT:END"
    assert frame.frame_id == "some-other-id"
    frame = updated_stack.frames[2]
    assert isinstance(frame, CancelPatternFlowStackFrame)
    assert frame.flow_id == "pattern_cancel_flow"
    assert frame.step_id == "1"
    assert frame.frame_id == "test_id"
    assert frame.canceled_name == "bar_flow"
