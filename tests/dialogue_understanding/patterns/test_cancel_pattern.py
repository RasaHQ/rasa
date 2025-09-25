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
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import AgentStackFrame
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


async def test_action_cancel_flow_cancels_agent_frame() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="user-1")
    agent_frame = AgentStackFrame(
        flow_id="bar_flow",
        step_id="2",
        frame_id="agent-1",
        agent_id="agent-x",
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="cancel-1",
        step_id="1",
        canceled_name="bar_flow",
        canceled_frames=["agent-1"],
    )
    stack = DialogueStack(frames=[user_frame, agent_frame, cancel_frame])

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

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    assert len(updated_stack.frames) == 2
    # user frame remains unchanged
    frame0 = updated_stack.frames[0]
    assert isinstance(frame0, UserFlowStackFrame)
    assert frame0.flow_id == "foo_flow"
    assert frame0.step_id == "1"
    assert frame0.frame_id == "user-1"
    # agent frame is removed, cancel frame stays
    frame1 = updated_stack.frames[1]
    assert isinstance(frame1, CancelPatternFlowStackFrame)
    assert frame1.frame_id == "cancel-1"


async def test_action_cancel_flow_cancels_user_and_agent_frames() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="user-1")
    agent_frame = AgentStackFrame(
        flow_id="bar_flow",
        step_id="2",
        frame_id="agent-1",
        agent_id="agent-x",
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="cancel-1",
        step_id="1",
        canceled_name="mixed",
        canceled_frames=["user-1", "agent-1"],
    )
    stack = DialogueStack(frames=[user_frame, agent_frame, cancel_frame])

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

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    # agent removed, user set to NEXT:END, cancel frame remains
    assert len(updated_stack.frames) == 2
    frame0 = updated_stack.frames[0]
    assert isinstance(frame0, UserFlowStackFrame)
    assert frame0.flow_id == "foo_flow"
    assert frame0.step_id == "NEXT:END"
    assert frame0.frame_id == "user-1"
    frame1 = updated_stack.frames[1]
    assert isinstance(frame1, CancelPatternFlowStackFrame)
    assert frame1.frame_id == "cancel-1"


async def test_action_cancel_flow_multiple_agent_frames_removed() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="user-1")
    agent_frame1 = AgentStackFrame(
        flow_id="bar_flow",
        step_id="2",
        frame_id="agent-1",
        agent_id="agent-x",
    )
    agent_frame2 = AgentStackFrame(
        flow_id="baz_flow",
        step_id="3",
        frame_id="agent-2",
        agent_id="agent-y",
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="cancel-1",
        step_id="1",
        canceled_name="agents",
        canceled_frames=["agent-1", "agent-2"],
    )
    stack = DialogueStack(frames=[user_frame, agent_frame1, agent_frame2, cancel_frame])

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

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    # both agents removed; user and cancel remain
    assert len(updated_stack.frames) == 2
    frame0 = updated_stack.frames[0]
    assert isinstance(frame0, UserFlowStackFrame)
    assert frame0.frame_id == "user-1"
    frame1 = updated_stack.frames[1]
    assert isinstance(frame1, CancelPatternFlowStackFrame)
    assert frame1.frame_id == "cancel-1"


async def test_action_cancel_flow_does_not_remove_uncanceled_agent() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="1", frame_id="user-1")
    agent_frame1 = AgentStackFrame(
        flow_id="bar_flow",
        step_id="2",
        frame_id="agent-1",
        agent_id="agent-x",
    )
    agent_frame2 = AgentStackFrame(
        flow_id="baz_flow",
        step_id="3",
        frame_id="agent-2",
        agent_id="agent-y",
    )
    cancel_frame = CancelPatternFlowStackFrame(
        frame_id="cancel-1",
        step_id="1",
        canceled_name="agents",
        canceled_frames=["agent-1"],
    )
    stack = DialogueStack(frames=[user_frame, agent_frame1, agent_frame2, cancel_frame])

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

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(event.update)

    # only agent-1 removed; agent-2 and others remain
    assert len(updated_stack.frames) == 3
    assert isinstance(updated_stack.frames[0], UserFlowStackFrame)
    # remaining agent frame should still be present
    remaining_types = [type(f).__name__ for f in updated_stack.frames]
    assert "AgentStackFrame" in remaining_types
    ids = [getattr(f, "frame_id", None) for f in updated_stack.frames]
    assert "agent-2" in ids and "agent-1" not in ids
