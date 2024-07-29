from rasa.dialogue_understanding.patterns.session_start import (
    SessionStartPatternFlowStackFrame,
)


async def test_chitchat_error_pattern_flow_stack_frame_type() -> None:
    frame = SessionStartPatternFlowStackFrame()
    assert frame.type() == "pattern_session_start"


async def test_chitchat_pattern_flow_stack_frame_from_dict() -> None:
    frame = SessionStartPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_session_start"
    assert frame.type() == "pattern_session_start"
