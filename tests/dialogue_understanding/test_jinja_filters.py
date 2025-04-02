import pytest

from rasa.dialogue_understanding.generator._jinja_filters import to_json_escaped_string


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ("Hello\nWorld", '"Hello\\nWorld"'),
        ("Tab\tSeparated", '"Tab\\tSeparated"'),
        ('He said "Hello"', '"He said \\"Hello\\""'),
        ("Normal text", '"Normal text"'),
        ("Umlauts äöüß", '"Umlauts äöüß"'),
        (
            'Mixed "quotes" and \n newlines \t tabs',
            '"Mixed \\"quotes\\" and \\n newlines \\t tabs"',
        ),
    ],
)
def test_to_json_escaped_string(input_str, expected_output):
    assert to_json_escaped_string(input_str) == expected_output
