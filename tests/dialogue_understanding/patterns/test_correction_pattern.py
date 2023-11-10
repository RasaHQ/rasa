from pytest import CaptureFixture

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.correction import (
    ActionCorrectFlowSlot,
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import SlotSet
from rasa.shared.core.trackers import DialogueStateTracker


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
        evts=[stack.persist_as_event()],
    )
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
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="some_step_id")
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
    stack = DialogueStack(frames=[user_frame, collect_info_frame, correction_frame])
    tracker = DialogueStateTracker.from_events(
        "test",
        domain=domain,
        slots=domain.slots,
        evts=[stack.persist_as_event()],
    )
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert len(events) == 2
    stack_event = events[0]
    assert isinstance(stack_event, SlotSet)
    assert stack_event.key == DIALOGUE_STACK_SLOT
    assert len(stack_event.value) == 3
    assert stack_event.value[0]["type"] == UserFlowStackFrame.type()
    assert stack_event.value[0]["flow_id"] == "foo_flow"
    assert stack_event.value[0]["step_id"] == "START"
    assert (
        stack_event.value[1]["type"] == CollectInformationPatternFlowStackFrame.type()
    )
    assert stack_event.value[1]["flow_id"] == "pattern_collect_information"
    assert stack_event.value[1]["step_id"] == "NEXT:END"
    assert stack_event.value[2]["type"] == CorrectionPatternFlowStackFrame.type()
    assert stack_event.value[2]["flow_id"] == "pattern_correction"
    assert stack_event.value[2]["step_id"] == "1"
    assert stack_event.value[2]["corrected_slots"] == {"foo": "bar"}
    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"


async def test_action_correct_flow_slot() -> None:
    domain = Domain.empty()
    user_frame = UserFlowStackFrame(flow_id="foo_flow", step_id="some_step_id")
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
        evts=[stack.persist_as_event()],
    )
    action = ActionCorrectFlowSlot()
    events = await action.run(
        CollectingOutputChannel(),
        TemplatedNaturalLanguageGenerator({}),
        tracker,
        domain,
    )
    assert len(events) == 2
    stack_event = events[0]
    assert isinstance(stack_event, SlotSet)
    assert stack_event.key == DIALOGUE_STACK_SLOT
    assert len(stack_event.value) == 3
    assert stack_event.value[0]["type"] == UserFlowStackFrame.type()
    assert stack_event.value[0]["flow_id"] == "foo_flow"
    assert stack_event.value[0]["step_id"] == "NEXT:ask_some_slot"
    assert (
        stack_event.value[1]["type"] == CollectInformationPatternFlowStackFrame.type()
    )
    assert stack_event.value[1]["flow_id"] == "pattern_collect_information"
    assert stack_event.value[1]["step_id"] == "NEXT:END"
    assert stack_event.value[2]["type"] == CorrectionPatternFlowStackFrame.type()
    assert stack_event.value[2]["flow_id"] == "pattern_correction"
    assert stack_event.value[2]["step_id"] == "1"
    assert stack_event.value[2]["corrected_slots"] == {"foo": "bar"}
    correction_slot_set_event = events[1]
    assert isinstance(correction_slot_set_event, SlotSet)
    assert correction_slot_set_event.key == "foo"
    assert correction_slot_set_event.value == "bar"
