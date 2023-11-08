import pytest
from rasa.core.actions.action_trigger_flow import ActionTriggerFlow
from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import ActiveLoop, FlowInterrupted, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


async def test_action_trigger_flow():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(channel, nlg, tracker, Domain.empty())
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, SlotSet)
    assert event.key == DIALOGUE_STACK_SLOT
    assert len(event.value) == 1
    assert event.value[0]["type"] == UserFlowStackFrame.type()
    assert event.value[0]["flow_id"] == "foo"
    assert event.value[0]["frame_type"] == FlowStackFrameType.REGULAR.value


async def test_action_trigger_flow_with_slots():
    tracker = DialogueStateTracker.from_events("test", [])
    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})
    events = await action.run(
        channel, nlg, tracker, Domain.empty(), metadata={"slots": {"foo": "bar"}}
    )

    event = events[0]
    assert isinstance(event, SlotSet)
    assert event.key == DIALOGUE_STACK_SLOT
    assert len(event.value) == 1
    assert event.value[0]["type"] == UserFlowStackFrame.type()
    assert event.value[0]["flow_id"] == "foo"

    assert len(events) == 2
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
    tracker = DialogueStateTracker.from_events("test", [stack.persist_as_event()])

    action = ActionTriggerFlow("flow_foo")
    channel = CollectingOutputChannel()
    nlg = TemplatedNaturalLanguageGenerator({})

    events = await action.run(channel, nlg, tracker, Domain.empty())

    assert len(events) == 2

    # The first event is a FlowInterrupted event
    flow_interrupted = events[0]
    assert flow_interrupted == FlowInterrupted("my_flow", "collect_bar")

    stack_event = events[1]
    assert isinstance(stack_event, SlotSet)
    assert stack_event.key == DIALOGUE_STACK_SLOT
    assert len(stack_event.value) == 2
    assert stack_event.value[1]["type"] == UserFlowStackFrame.type()
    assert stack_event.value[1]["flow_id"] == "foo"
    assert stack_event.value[1]["frame_type"] == FlowStackFrameType.INTERRUPT.value
