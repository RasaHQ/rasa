from rasa.shared.nlu.training_data.util import has_string_escape_chars
import pytest


@pytest.mark.parametrize(
    "s, result",
    [
        ("Hey,\nmy name is Christof", True),
        ("Howdy!", False),
        ("A\tB", True),
        ("Hey,\rmy name is Thomas", True),
        ("Hey, my name is Thomas", False),
    ],
)
def test_has_string_escape_chars(s, result):
    assert has_string_escape_chars(s) == result
