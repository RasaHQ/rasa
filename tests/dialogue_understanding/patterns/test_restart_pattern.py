from rasa.dialogue_understanding.patterns.restart import RestartPatternFlowStackFrame


async def test_chitchat_error_pattern_flow_stack_frame_type() -> None:
    frame = RestartPatternFlowStackFrame()
    assert frame.type() == "pattern_restart"


async def test_chitchat_pattern_flow_stack_frame_from_dict() -> None:
    frame = RestartPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_restart"
    assert frame.type() == "pattern_restart"
