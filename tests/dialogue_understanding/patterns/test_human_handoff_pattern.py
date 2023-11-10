from rasa.dialogue_understanding.patterns.human_handoff import (
    HumanHandoffPatternFlowStackFrame,
)


async def test_internal_error_pattern_flow_stack_frame_type() -> None:
    frame = HumanHandoffPatternFlowStackFrame(flow_id="test")
    assert frame.type() == "pattern_human_handoff"


async def test_internal_error_pattern_flow_stack_frame_from_dict() -> None:
    frame = HumanHandoffPatternFlowStackFrame.from_dict(
        {"frame_id": "test_id", "step_id": "test_step_id"}
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_human_handoff"
    assert frame.type() == "pattern_human_handoff"
