from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame


def test_cancel_pattern_flow_stack_frame_type():
    frame = CancelPatternFlowStackFrame()
    assert frame.type() == "pattern_cancel_flow"


def test_cancel_pattern_flow_stack_frame_from_dict():
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
