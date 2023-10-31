from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.flows.flow import SlotRejection


def test_collect_information_pattern_flow_stack_frame_type():
    frame = CollectInformationPatternFlowStackFrame()
    assert frame.type() == "pattern_collect_information"


def test_collect_information_pattern_flow_stack_frame_from_dict():
    frame = CollectInformationPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "test_step_id",
            "utter": "test_utter",
            "collect": "test_slot",
            "rejections": [{"if": "condition", "utter": "sample_a"}],
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "test_step_id"
    assert frame.utter == "test_utter"
    assert frame.collect == "test_slot"
    assert frame.rejections == [
        SlotRejection.from_dict({"if": "condition", "utter": "sample_a"})
    ]
    assert frame.flow_id == "pattern_collect_information"
    assert frame.type() == "pattern_collect_information"


def test_collect_information_pattern_flow_stack_context_as_dict():
    frame = CollectInformationPatternFlowStackFrame(
        frame_id="test_id",
        step_id="test_step_id",
        utter="test_utter",
        collect="test_slot",
        rejections=[SlotRejection.from_dict({"if": "condition", "utter": "sample_a"})],
    )
    assert frame.context_as_dict([]) == {
        "frame_id": "test_id",
        "step_id": "test_step_id",
        "utter": "test_utter",
        "collect": "test_slot",
        "rejections": [{"if": "condition", "utter": "sample_a"}],
        "type": "pattern_collect_information",
        "flow_id": "pattern_collect_information",
    }


def test_collect_information_pattern_flow_stack_context_as_dict_with_frame():
    frame = CollectInformationPatternFlowStackFrame(
        frame_id="test_id",
        step_id="test_step_id",
        utter="test_utter",
        collect="test_slot",
        rejections=[SlotRejection.from_dict({"if": "condition", "utter": "sample_a"})],
    )
    assert frame.context_as_dict(
        [
            UserFlowStackFrame(
                flow_id="user_flow_id",
                step_id="some_step",
                frame_id="some_frame",
            )
        ]
    ) == {
        "frame_id": "test_id",
        "utter": "test_utter",
        "collect": "test_slot",
        "rejections": [{"if": "condition", "utter": "sample_a"}],
        "type": "flow",
        "flow_id": "user_flow_id",
        "step_id": "some_step",
        "frame_id": "some_frame",
        "frame_type": "regular",
    }
