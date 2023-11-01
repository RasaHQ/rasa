from pytest import CaptureFixture

from rasa.core.channels import CollectingOutputChannel
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.correction import (
    ActionCorrectFlowSlot,
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import (
    DialogueStack,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.domain import Domain
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
