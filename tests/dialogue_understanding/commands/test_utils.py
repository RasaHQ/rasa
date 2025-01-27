from typing import List

import pytest

from rasa.dialogue_understanding.commands.utils import (
    clean_extracted_value,
    extract_cleaned_options,
    is_none_value,
)


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
    ],
)
def test_extract_cleaned_options(input_value: str, expected_output: List[str]):
    # When
    cleaned_options = extract_cleaned_options(input_value)
    # Then
    assert cleaned_options == expected_output
