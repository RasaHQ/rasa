from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)


def test_internal_error_pattern_flow_stack_frame_type():
    frame = InternalErrorPatternFlowStackFrame(flow_id="test")
    assert frame.type() == "pattern_internal_error"


def test_internal_error_pattern_flow_stack_frame_from_dict():
    frame = InternalErrorPatternFlowStackFrame.from_dict(
        {"frame_id": "test_id", "step_id": "test_step_id"}
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_internal_error"
    assert frame.type() == "pattern_internal_error"
