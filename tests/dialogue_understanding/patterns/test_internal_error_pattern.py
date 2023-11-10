from rasa.dialogue_understanding.patterns.internal_error import (
    InternalErrorPatternFlowStackFrame,
)


def test_internal_error_pattern_flow_stack_frame_type() -> None:
    frame = InternalErrorPatternFlowStackFrame(flow_id="test")
    assert frame.type() == "pattern_internal_error"


def test_internal_error_pattern_flow_stack_frame_message() -> None:
    frame = InternalErrorPatternFlowStackFrame(flow_id="test", message="test message")
    assert frame.message == "test message"


def test_internal_error_pattern_flow_stack_frame_from_dict() -> None:
    frame = InternalErrorPatternFlowStackFrame.from_dict(
        {"frame_id": "test_id", "step_id": "test_step_id", "message": "test message"}
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_internal_error"
    assert frame.message == "test message"
    assert frame.type() == "pattern_internal_error"
