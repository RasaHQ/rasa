from typing import Text, Dict

import pytest

from rasa.core.events import UserUttered
from rasa.core.training.dsl import EndToEndReader


@pytest.mark.parametrize(
    "line, expected",
    [
        (" greet: hi", {"intent": "greet", "true_intent": "greet", "text": "hi"}),
        (
            " greet: /greet",
            {
                "intent": "greet",
                "true_intent": "greet",
                "text": "/greet",
                "entities": [],
            },
        ),
        (
            'greet: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            'greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"entity": "test", "start": 6, "end": 22, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
        (
            "mood_great: [great](feeling)",
            {
                "intent": "mood_great",
                "entities": [
                    {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                ],
                "true_intent": "mood_great",
                "text": "great",
            },
        ),
        (
            'form: greet{"test": "test"}: /greet{"test": "test"}',
            {
                "intent": "greet",
                "entities": [
                    {"end": 22, "entity": "test", "start": 6, "value": "test"}
                ],
                "true_intent": "greet",
                "text": '/greet{"test": "test"}',
            },
        ),
    ],
)
def test_e2e_parsing(line: Text, expected: Dict):
    reader = EndToEndReader()
    actual = reader._parse_item(line)

    assert actual.as_dict() == expected


@pytest.mark.parametrize(
    "parse_data, expected_story_string",
    [
        (
            {
                "text": "/simple",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: /simple",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [
                        {"start": 0, "end": 5, "value": "great", "entity": "feeling"}
                    ],
                },
            },
            "simple: [great](feeling)",
        ),
        (
            {
                "text": "great",
                "parse_data": {
                    "intent": {"confidence": 1.0, "name": "simple"},
                    "entities": [],
                },
            },
            "simple: great",
        ),
    ],
)
def test_user_uttered_to_e2e(parse_data: Dict, expected_story_string: Text):
    event = UserUttered.from_story_string("user", parse_data)[0]

    assert isinstance(event, UserUttered)
    assert event.as_story_string(e2e=True) == expected_story_string


@pytest.mark.parametrize("line", [" greet{: hi"])
def test_invalid_end_to_end_format(line: Text):
    reader = EndToEndReader()

    with pytest.raises(ValueError):
        # noinspection PyProtectedMember
        _ = reader._parse_item(line)
