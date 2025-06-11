from typing import Any, List

import pytest

from rasa.dialogue_understanding.commands.utils import (
    clean_extracted_value,
    extract_cleaned_options,
    find_default_flows_collecting_slot,
    initialize_pattern_validate_slot,
    is_none_value,
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
