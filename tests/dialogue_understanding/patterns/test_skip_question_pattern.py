from rasa.dialogue_understanding.patterns.skip_question import (
    SkipQuestionPatternFlowStackFrame,
)


async def test_skip_question_error_pattern_flow_stack_frame_type() -> None:
    frame = SkipQuestionPatternFlowStackFrame()
    assert frame.type() == "pattern_skip_question"


async def test_skip_question_pattern_flow_stack_frame_from_dict() -> None:
    frame = SkipQuestionPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_skip_question"
    assert frame.type() == "pattern_skip_question"
