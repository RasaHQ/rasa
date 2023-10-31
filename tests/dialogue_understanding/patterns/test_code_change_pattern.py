from rasa.dialogue_understanding.patterns.code_change import (
    CodeChangeFlowStackFrame,
)


def test_code_change_error_pattern_flow_stack_frame_type():
    frame = CodeChangeFlowStackFrame()
    assert frame.type() == "pattern_code_change"


def test_code_change_pattern_flow_stack_frame_from_dict():
    frame = CodeChangeFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_code_change"
    assert frame.type() == "pattern_code_change"
