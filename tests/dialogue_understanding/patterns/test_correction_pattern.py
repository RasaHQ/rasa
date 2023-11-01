from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
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
