from rasa.dialogue_understanding.patterns.handle_digressions import (
    HandleDigressionsPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame


def test_handle_digressions_pattern_flow_stack_frame_type():
    frame = HandleDigressionsPatternFlowStackFrame()
    assert frame.type() == "pattern_handle_digressions"


def test_handle_digressions_pattern_flow_stack_frame_from_dict():
    frame = HandleDigressionsPatternFlowStackFrame.from_dict(
        {
            "frame_id": "test_id",
            "step_id": "START",
            "interrupted_step_id": "test_interrupted_step_id",
            "interrupted_flow_id": "test_interrupted_flow_id",
            "interrupting_flow_id": "test_interrupting_flow_id",
            "ask_confirm_digressions": ["test_ask_confirm_digressions"],
            "block_digressions": ["test_block_digressions"],
        }
    )
    assert frame.frame_id == "test_id"
    assert frame.step_id == "START"
    assert frame.interrupted_step_id == "test_interrupted_step_id"
    assert frame.interrupted_flow_id == "test_interrupted_flow_id"
    assert frame.interrupting_flow_id == "test_interrupting_flow_id"
    assert frame.ask_confirm_digressions == {"test_ask_confirm_digressions"}
    assert frame.block_digressions == {"test_block_digressions"}
    assert frame.flow_id == "pattern_handle_digressions"
    assert frame.type() == "pattern_handle_digressions"


def test_handle_digressions_pattern_flow_stack_context_as_dict_empty():
    frame = HandleDigressionsPatternFlowStackFrame(
        frame_id="test_id",
        step_id="START",
        interrupted_step_id="test_interrupted_step_id",
        interrupted_flow_id="test_interrupted_flow_id",
        interrupting_flow_id="test_interrupting_flow_id",
        ask_confirm_digressions={"test_ask_confirm_digressions"},
        block_digressions={"test_block_digressions"},
    )
    assert frame.context_as_dict([]) == {
        "frame_id": "test_id",
        "step_id": "START",
        "interrupted_step_id": "test_interrupted_step_id",
        "interrupted_flow_id": "test_interrupted_flow_id",
        "interrupting_flow_id": "test_interrupting_flow_id",
        "ask_confirm_digressions": ["test_ask_confirm_digressions"],
        "block_digressions": ["test_block_digressions"],
        "type": "pattern_handle_digressions",
        "flow_id": "pattern_handle_digressions",
    }


def test_handle_digressions_pattern_flow_stack_context_as_dict():
    frame = HandleDigressionsPatternFlowStackFrame(
        frame_id="test_id",
        step_id="START",
        interrupted_step_id="test_interrupted_step_id",
        interrupted_flow_id="test_interrupted_flow_id",
        interrupting_flow_id="test_interrupting_flow_id",
        ask_confirm_digressions={"test_ask_confirm_digressions"},
        block_digressions={"test_block_digressions"},
    )
    assert frame.context_as_dict(
        [
            UserFlowStackFrame(
                flow_id="test_interrupted_flow_id",
                step_id="test_interrupted_step_id",
                frame_id="some_frame",
            )
        ]
    ) == {
        "frame_id": "test_id",
        "step_id": "START",
        "interrupted_step_id": "test_interrupted_step_id",
        "interrupted_flow_id": "test_interrupted_flow_id",
        "interrupting_flow_id": "test_interrupting_flow_id",
        "ask_confirm_digressions": ["test_ask_confirm_digressions"],
        "block_digressions": ["test_block_digressions"],
        "type": "pattern_handle_digressions",
        "flow_id": "pattern_handle_digressions",
    }
