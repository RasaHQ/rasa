from rasa.dialogue_understanding.patterns.clarify import ClarifyPatternFlowStackFrame


async def test_clarify_pattern_flow_stack_frame_type() -> None:
    frame = ClarifyPatternFlowStackFrame()
    assert frame.type() == "pattern_clarification"


async def test_clarify_pattern_flow_stack_frame_from_dict() -> None:
    frame = ClarifyPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "names": "foo",
            "clarification_options": "",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.names == "foo"
    assert frame.clarification_options == ""
    assert frame.flow_id == "pattern_clarification"
    assert frame.type() == "pattern_clarification"
