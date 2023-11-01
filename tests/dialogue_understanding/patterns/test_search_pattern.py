from rasa.dialogue_understanding.patterns.search import SearchPatternFlowStackFrame


async def test_search_pattern_flow_stack_frame_type() -> None:
    frame = SearchPatternFlowStackFrame(flow_id="test")
    assert frame.type() == "pattern_search"


async def test_search_pattern_flow_stack_frame_from_dict() -> None:
    frame = SearchPatternFlowStackFrame.from_dict(
        {"frame_id": "test_id", "step_id": "test_step_id"}
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_search"
    assert frame.type() == "pattern_search"
