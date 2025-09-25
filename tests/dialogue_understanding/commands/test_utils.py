from typing import Any, List, Optional

import pytest

from rasa.dialogue_understanding.commands.utils import (
    clean_extracted_value,
    collect_frames_to_resume,
    extract_cleaned_options,
    find_default_flows_collecting_slot,
    initialize_pattern_validate_slot,
    is_none_value,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    UserFlowStackFrame,
)
from rasa.shared.constants import REFILL_UTTER, REJECTIONS
from rasa.shared.core.slots import (
    AnySlot,
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    ListSlot,
    SlotRejection,
    StrictCategoricalSlot,
    TextSlot,
)
from rasa.shared.core.trackers import DialogueStateTracker
from tests.utilities import flows_from_str


@pytest.mark.parametrize(
    "input_value, expected_truthiness",
    [
        ("", True),
        (" ", False),
        ("none", False),
        ("some text", False),
        ("[missing information]", True),
        ("[missing]", True),
        ("None", True),
        ("undefined", True),
        ("null", True),
    ],
)
def test_is_none_value(
    input_value: str,
    expected_truthiness: bool,
):
    """Test that is_none_value returns True when the value is None."""
    assert is_none_value(input_value) == expected_truthiness


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("text", "text"),
        (" text ", "text"),
        ('"text"', "text"),
        ("'text'", "text"),
        ("' \"text' \"  ", "text"),
        ("", ""),
    ],
)
def test_clean_extracted_value(input_value: str, expected_output: str):
    """Test that clean_extracted_value removes
    the leading and trailing whitespaces.
    """
    # When
    cleaned_value = clean_extracted_value(input_value)
    # Then
    assert cleaned_value == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("", []),
        (" ", []),
        ("abc", ["abc"]),
        ("'abc'", ["abc"]),
        ('"abc",', ["abc"]),
        ("abc, def, ghi", ["abc", "def", "ghi"]),
        ("abc,def,ghi", ["abc", "def", "ghi"]),
        ("abc, def,", ["abc", "def"]),
        ("abc, def, ,", ["abc", "def"]),
        ("abc   , def   ", ["abc", "def"]),
        ("'abc', 'def', 'ghi'", ["abc", "def", "ghi"]),
        ('"abc", "def", "ghi"', ["abc", "def", "ghi"]),
        ("'abc', ''def'', 'ghi", ["abc", "def", "ghi"]),
        ("'abc', 'def', 'ghi", ["abc", "def", "ghi"]),
        ("abc def ghi", ["abc", "def", "ghi"]),
        ("'abc' 'def' 'ghi'", ["abc", "def", "ghi"]),
        ("'abc', 'def' 'ghi", ["abc", "def", "ghi"]),
        ('"abc", "def" "ghi"', ["abc", "def", "ghi"]),
        ("'abc'    'def' 'ghi     ", ["abc", "def", "ghi"]),
    ],
)
def test_extract_cleaned_options(input_value: str, expected_output: List[str]):
    # When
    cleaned_options = extract_cleaned_options(input_value)
    # Then
    assert cleaned_options == expected_output


@pytest.mark.parametrize(
    "slot_type",
    [
        (TextSlot),
        (FloatSlot),
        (CategoricalSlot),
        (AnySlot),
        (BooleanSlot),
        (ListSlot),
        (StrictCategoricalSlot),
    ],
)
def test_initialize_pattern_validate_slot_with_validation(slot_type: Any):
    """Test that the method creates a ValidateSlotPatternFlowStackFrame"""
    # Given
    slot = slot_type(
        name="test_slot",
        mappings=[],
        validation={
            REFILL_UTTER: "utter_test_slot",
            REJECTIONS: [{"if": "test_condition", "utter": "utter_invalid_test_slot"}],
        },
    )
    # When
    validate_frame = initialize_pattern_validate_slot(slot)
    # Then
    assert validate_frame is not None
    assert validate_frame.validate == "test_slot"
    assert validate_frame.refill_utter == "utter_test_slot"
    assert validate_frame.refill_action == "action_ask_test_slot"
    assert validate_frame.rejections == [
        SlotRejection(if_="test_condition", utter="utter_invalid_test_slot")
    ]
    assert validate_frame.flow_id == "pattern_validate_slot"
    assert validate_frame.type() == "pattern_validate_slot"


def test_initialize_pattern_validate_slot_without_validation():
    """Test that the method returns None when validation is not required"""
    # Given
    slot = TextSlot(
        name="test_slot",
        mappings=[],
    )
    # When
    validate_frame = initialize_pattern_validate_slot(slot)
    # Then
    assert validate_frame is None


@pytest.mark.parametrize(
    "slot_name, expected",
    [
        ("bar", ["pattern_completed"]),
        ("random_slot", []),
    ],
)
def test_find_default_flows_collecting_slot(
    slot_name: str, expected: List[str]
) -> None:
    all_flows = flows_from_str(
        """
        flows:
          flow1:
            description: "Flow 1"
            steps:
              - collect: foo
          pattern_completed:
            description: "Pattern Completed"
            steps:
              - collect: bar
        """
    )

    predicted_flows = find_default_flows_collecting_slot(slot_name, all_flows)
    assert predicted_flows == expected


def test_collect_frames_to_resume_simple():
    """Test the _collect_frames_to_resume helper method with a simple stack."""
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    frames_to_resume, user_frame_to_resume = collect_frames_to_resume(
        tracker.stack, "bar"
    )

    assert len(frames_to_resume) == 1
    assert frames_to_resume[0].flow_id == "bar"
    assert user_frame_to_resume is not None
    assert user_frame_to_resume.flow_id == "bar"


def test_collect_frames_to_resume_with_pattern_frames():
    """Test the _collect_frames_to_resume helper method with pattern frames."""
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "pattern_collect_information",
                    "flow_id": "pattern_collect_information",
                    "step_id": "START",
                    "frame_id": "pattern-frame",
                    "collect": "first_step",
                    "collect_action": "action_ask_first_step",
                    "utter": "utter_ask_first_step",
                },
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    frames_to_resume, user_frame_to_resume = collect_frames_to_resume(
        tracker.stack, "bar"
    )

    assert len(frames_to_resume) == 2
    assert frames_to_resume[0].flow_id == "bar"
    assert frames_to_resume[1].flow_id == "pattern_collect_information"
    assert user_frame_to_resume is not None
    assert user_frame_to_resume.flow_id == "bar"


def test_collect_frames_to_resume_flow_not_found():
    """Test the _collect_frames_to_resume helper method when flow is not found."""
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    frames_to_resume, user_frame_to_resume = collect_frames_to_resume(
        tracker.stack, "bar"
    )

    assert len(frames_to_resume) == 0
    assert user_frame_to_resume is None


def test_collect_frames_to_resume_with_interrupt_frames():
    """Test the _collect_frames_to_resume helper method with interrupt frames."""
    tracker = DialogueStateTracker.from_events("test", evts=[])
    tracker.update_stack(
        DialogueStack.from_dict(
            [
                {
                    "type": "flow",
                    "frame_type": "regular",
                    "flow_id": "bar",
                    "step_id": "first_step",
                    "frame_id": "bar-frame",
                },
                {
                    "type": "flow",
                    "frame_type": "interrupt",
                    "flow_id": "foo",
                    "step_id": "first_step",
                    "frame_id": "foo-frame",
                },
            ]
        )
    )

    frames_to_resume, user_frame_to_resume = collect_frames_to_resume(
        tracker.stack, "bar"
    )

    assert len(frames_to_resume) == 1
    assert frames_to_resume[0].flow_id == "bar"
    assert frames_to_resume[0].frame_type == "regular"
    assert user_frame_to_resume is not None
    assert user_frame_to_resume.flow_id == "bar"


@pytest.mark.parametrize(
    "initial_frames,expected_frame_count,expected_top_frame_frame_id",
    [
        ([], 0, None),
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="START", frame_id="user-frame-1"
                )
            ],
            1,
            "user-frame-1",
        ),
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="START", frame_id="user-frame-1"
                ),
                CollectInformationPatternFlowStackFrame(
                    frame_id="pattern-frame-1",
                    step_id="pattern_step",
                ),
            ],
            2,
            "pattern-frame-1",
        ),
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="START", frame_id="user-frame-1"
                ),
                ContinueInterruptedPatternFlowStackFrame(
                    frame_id="continue-pattern-frame",
                    step_id="continue_step",
                    interrupted_flow_names=["previous_flow"],
                    interrupted_flow_ids=["previous_flow_id"],
                    interrupted_flow_options="previous_flow_options",
                ),
            ],
            1,
            "user-frame-1",
        ),
        (
            [
                UserFlowStackFrame(
                    flow_id="test_flow", step_id="START", frame_id="user-frame-1"
                ),
                ContinueInterruptedPatternFlowStackFrame(
                    frame_id="pattern-frame-1",
                    step_id="pattern_step",
                    interrupted_flow_names=["previous_flow"],
                    interrupted_flow_ids=["previous_flow_id"],
                    interrupted_flow_options="previous_flow_options",
                ),
                CollectInformationPatternFlowStackFrame(
                    frame_id="pattern-frame-2",
                    step_id="pattern_step",
                ),
            ],
            1,
            "user-frame-1",
        ),
        (
            [
                ContinueInterruptedPatternFlowStackFrame(
                    frame_id="continue-pattern-frame-1",
                    step_id="continue_step_1",
                    interrupted_flow_names=["previous_flow_1"],
                    interrupted_flow_ids=["previous_flow_id_1"],
                    interrupted_flow_options="previous_flow_options_1",
                ),
                ContinueInterruptedPatternFlowStackFrame(
                    frame_id="continue-pattern-frame-2",
                    step_id="continue_step_2",
                    interrupted_flow_names=["previous_flow_2"],
                    interrupted_flow_ids=["previous_flow_id_2"],
                    interrupted_flow_options="previous_flow_options_2",
                ),
            ],
            0,
            None,
        ),
    ],
)
def test_remove_pattern_continue_interrupted_frames(
    initial_frames: List[DialogueStackFrame],
    expected_frame_count: int,
    expected_top_frame_frame_id: Optional[str],
):
    """Test removing pattern frames from various stack configurations."""
    from rasa.dialogue_understanding.commands.start_flow_command import (
        remove_pattern_continue_interrupted_frames,
    )

    stack = DialogueStack(frames=initial_frames)
    result = remove_pattern_continue_interrupted_frames(stack)

    assert len(result.frames) == expected_frame_count

    if expected_top_frame_frame_id:
        assert result.frames[-1].frame_id == expected_top_frame_frame_id


def test_remove_pattern_continue_interrupted_frames_preserves_original_stack():
    """Test that the original stack is not modified."""
    from rasa.dialogue_understanding.commands.start_flow_command import (
        remove_pattern_continue_interrupted_frames,
    )

    user_frame = UserFlowStackFrame(
        flow_id="test_flow", step_id="START", frame_id="user-frame-1"
    )
    pattern_frame = ContinueInterruptedPatternFlowStackFrame(
        frame_id="pattern-frame-1",
        step_id="pattern_step",
        interrupted_flow_names=["previous_flow"],
        interrupted_flow_ids=["previous_flow_id"],
        interrupted_flow_options="previous_flow_options",
    )

    original_stack = DialogueStack(frames=[user_frame, pattern_frame])
    stack_copy = original_stack.copy()

    result = remove_pattern_continue_interrupted_frames(stack_copy)

    # Original stack should remain unchanged
    assert len(original_stack.frames) == 2
    assert original_stack.frames[1] == pattern_frame

    # Result should have pattern frame removed
    assert len(result.frames) == 1
    assert result.frames[0] == user_frame


def test_remove_pattern_continue_interrupted_frames_complex_stack():
    """Test removing pattern frames from a complex stack with multiple frame types."""
    from rasa.dialogue_understanding.commands.start_flow_command import (
        remove_pattern_continue_interrupted_frames,
    )

    user_frame_1 = UserFlowStackFrame(
        flow_id="flow_1", step_id="START", frame_id="user-frame-1"
    )
    user_frame_2 = UserFlowStackFrame(
        flow_id="flow_2", step_id="START", frame_id="user-frame-2"
    )
    agent_frame = AgentStackFrame(
        frame_id="agent-frame-1",
        state=AgentState.WAITING_FOR_INPUT,
        agent_id="test_agent",
        flow_id="flow_2",
    )
    pattern_frame_1 = ContinueInterruptedPatternFlowStackFrame(
        frame_id="pattern-frame-1",
        step_id="pattern_step_1",
        interrupted_flow_names=["previous_flow_1"],
        interrupted_flow_ids=["previous_flow_id_1"],
        interrupted_flow_options="previous_flow_options_1",
    )
    pattern_frame_2 = CollectInformationPatternFlowStackFrame(
        frame_id="pattern-frame-2",
        step_id="pattern_step_2",
    )

    stack = DialogueStack(
        frames=[
            user_frame_1,
            user_frame_2,
            agent_frame,
            pattern_frame_1,
            pattern_frame_2,
        ]
    )

    result = remove_pattern_continue_interrupted_frames(stack)

    assert len(result.frames) == 3
    assert result.frames[0] == user_frame_1
    assert result.frames[1] == user_frame_2
    assert result.frames[2] == agent_frame
    assert result.frames[2].frame_id == "agent-frame-1"
