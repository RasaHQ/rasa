from typing import Any, List
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.dialogue_understanding.commands.set_slot_command import SetSlotExtractor
from rasa.dialogue_understanding.commands.utils import (
    clean_extracted_value,
    create_validate_frames_from_slot_set_events,
    extract_cleaned_options,
    initialize_pattern_validate_slot,
    is_none_value,
)
from rasa.dialogue_understanding.patterns.correction import (
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.constants import REFILL_UTTER, REJECTIONS
from rasa.shared.core.events import SlotSet
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


def test_create_validate_frames_from_slot_set_events_does_not_modify_corrected_slots(
    monkeypatch: MonkeyPatch,
):
    """Test that method validate_frames_from_slot_set_events modifies corrected slots"""
    # Given
    events = [SlotSet(key="test_slot", value="invalid")]
    tracker = DialogueStateTracker.from_events(
        "test",
        [],
        slots=[
            TextSlot("test_slot2", mappings=[]),
            TextSlot(
                name="test_slot",
                mappings=[],
                validation={
                    REFILL_UTTER: "utter_test_slot",
                    REJECTIONS: [
                        {"if": "test condition", "utter": "utter_invalid_test_slot"}
                    ],
                },
            ),
        ],
    )
    mock_update_corrected_slots_in_correction_frame = MagicMock()
    monkeypatch.setattr(
        "rasa.dialogue_understanding.commands.utils.update_corrected_slots_in_correction_frame",
        mock_update_corrected_slots_in_correction_frame,
    )
    # When
    tracker, validate_frames = create_validate_frames_from_slot_set_events(
        tracker, events, []
    )
    # Then
    assert len(validate_frames) == 1
    assert isinstance(validate_frames[0], ValidateSlotPatternFlowStackFrame)
    assert mock_update_corrected_slots_in_correction_frame.call_count == 0


def test_create_validate_frames_from_slot_set_events_modifies_corrected_slots():
    events = [
        SlotSet(key="test_slot", value="valid"),
        SlotSet(key="test_slot_2", value="valid"),
    ]

    tracker = DialogueStateTracker.from_events(
        "default",
        [],
        [
            TextSlot(
                name="test_slot",
                mappings=[],
                validation={
                    REFILL_UTTER: "utter_test_slot",
                    REJECTIONS: [
                        {
                            "if": "test_slot == invalid",
                            "utter": "utter_invalid_test_slot",
                        }
                    ],
                },
            ),
            TextSlot(
                name="test_slot_2",
                mappings=[],
                validation={
                    REFILL_UTTER: "utter_test_slot",
                    REJECTIONS: [
                        {
                            "if": "test_slot_2 == invalid",
                            "utter": "utter_invalid_test_slot",
                        }
                    ],
                },
            ),
        ],
    )
    correction_frame = CorrectionPatternFlowStackFrame(
        corrected_slots={
            "test_slot": {"value": "invalid", "filled_by": SetSlotExtractor.LLM.value},
            "test_slot_2": {"value": "valid", "filled_by": SetSlotExtractor.LLM.value},
        },
        new_slot_values=["invalid", "valid"],
    )
    tracker.update_stack(
        DialogueStack(
            frames=[
                UserFlowStackFrame(flow_id="foo", step_id="0_collect_foo_slot_a"),
                correction_frame,
            ]
        )
    )
    tracker.update_with_events(events)
    tracker, frames = create_validate_frames_from_slot_set_events(
        tracker, events, should_break=True, update_corrected_slots=True
    )

    assert len(frames) == 2
    assert isinstance(frames[0], ValidateSlotPatternFlowStackFrame)
    assert isinstance(frames[1], ValidateSlotPatternFlowStackFrame)
    top_frame = tracker.stack.top()
    assert top_frame != correction_frame
    assert "test_slot" not in top_frame.corrected_slots.keys()
    assert "invalid" not in top_frame.new_slot_values
