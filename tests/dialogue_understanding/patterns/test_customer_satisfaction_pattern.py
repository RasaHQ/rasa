from rasa.dialogue_understanding.patterns.customer_satisfaction import (
    CustomerSatisfactionPatternFlowStackFrame,
)


def test_customer_satisfaction_pattern_flow_stack_frame_type() -> None:
    frame = CustomerSatisfactionPatternFlowStackFrame()
    assert frame.type() == "pattern_customer_satisfaction"


def test_customer_satisfaction_pattern_flow_stack_frame_from_dict() -> None:
    frame = CustomerSatisfactionPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_customer_satisfaction"
    assert frame.type() == "pattern_customer_satisfaction"
