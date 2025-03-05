from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.shared.constants import REFILL_UTTER, REJECTIONS
from rasa.shared.core.slots import SlotRejection


async def test_validate_slot_pattern_flow_stack_frame_type() -> None:
    """Test that the type of the frame is 'pattern_validate_slot'."""
    frame = ValidateSlotPatternFlowStackFrame()
    assert frame.type() == "pattern_validate_slot"


async def test_validate_slot_pattern_flow_stack_frame_from_dict() -> None:
    """Test that the frame is created from a dictionary."""
    frame = ValidateSlotPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            REFILL_UTTER: "utter_ask_test_slot",
            "refill_action": "action_ask_test_slot",
            "validate": "test_slot",
            REJECTIONS: [{"if": "condition", "utter": "utter_invalid_test_slot"}],
        }
    )

    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.flow_id == "pattern_validate_slot"
    assert frame.type() == "pattern_validate_slot"
    assert frame.refill_utter == "utter_ask_test_slot"
    assert frame.refill_action == "action_ask_test_slot"
    assert frame.validate == "test_slot"
    assert frame.rejections == [
        SlotRejection.from_dict({"if": "condition", "utter": "utter_invalid_test_slot"})
    ]
