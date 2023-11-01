from rasa.dialogue_understanding.patterns.chitchat import (
    ChitchatPatternFlowStackFrame,
)


async def test_chitchat_error_pattern_flow_stack_frame_type() -> None:
    frame = ChitchatPatternFlowStackFrame()
    assert frame.type() == "pattern_chitchat"


async def test_chitchat_pattern_flow_stack_frame_from_dict() -> None:
    frame = ChitchatPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_chitchat"
    assert frame.type() == "pattern_chitchat"
