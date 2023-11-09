import pytest
from rasa.core.actions.action_trigger_flow import ActionTriggerFlow
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActiveLoop, DialogueStackUpdated, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


async def test_action_trigger_flow():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(channel, nlg, tracker, Domain.empty())
    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo"
    assert frame.frame_type == FlowStackFrameType.REGULAR.value


async def test_action_trigger_flow_with_slots():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(
        channel, nlg, tracker, Domain.empty(), metadata={"slots": {"foo": "bar"}}
    )

    assert len(events) == 2

    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 1

    frame = updated_stack.frames[0]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo"

    event = events[1]
    assert isinstance(event, SlotSet)
    assert event.key == "foo"
    assert event.value == "bar"


async def test_action_trigger_fails_if_name_is_invalid():
    with pytest.raises(ValueError):
        ActionTriggerFlow("foo")


async def test_action_trigger_ends_an_active_loop_on_the_tracker():
    tracker = DialogueStateTracker.from_events("test", [ActiveLoop("loop_foo")])
    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(channel, nlg, tracker, Domain.empty())

    assert len(events) == 2
    assert isinstance(events[1], ActiveLoop)
    assert events[1].name is None


async def test_action_trigger_uses_interrupt_flow_type_if_stack_already_contains_flow():
    user_frame = UserFlowStackFrame(
        flow_id="my_flow", step_id="collect_bar", frame_id="some-frame-id"
    )
    stack = DialogueStack(frames=[user_frame])
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(stack)

    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})

    events = await action.run(channel, nlg, tracker, Domain.empty())

    assert len(events) == 1
    dialogue_stack_event = events[0]
    assert isinstance(dialogue_stack_event, DialogueStackUpdated)

    updated_stack = tracker.stack.update_from_patch(dialogue_stack_event.update)

    assert len(updated_stack.frames) == 2

    frame = updated_stack.frames[1]
    assert isinstance(frame, UserFlowStackFrame)
    assert frame.flow_id == "foo"
    assert frame.frame_type == FlowStackFrameType.INTERRUPT.value
