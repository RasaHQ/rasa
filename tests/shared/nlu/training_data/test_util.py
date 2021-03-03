from rasa.shared.nlu.training_data.util import has_string_escape_chars
import pytest
from typing import Text


@pytest.mark.parametrize(
    "s, has_escaped_char",
    [
        ("Hey,\nmy name is Christof", True),
        ("Howdy!", False),
        ("A\tB", True),
        ("Hey,\rmy name is Thomas", True),
        ("Hey, my name is Thomas", False),
        ("Hey,\nI\ncan\nwrite\nmany\nlines.", True),
    ],
)
def test_has_string_escape_chars(s: Text, has_escaped_char: bool):
    assert has_string_escape_chars(s) == has_escaped_char
