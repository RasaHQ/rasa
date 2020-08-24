from typing import Any, Text, Dict, List

import pytest

from rasa.nlu.constants import TEXT
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.training_data.formats import MarkdownReader


@pytest.mark.parametrize(
    "text, tags, confidences, expected_entities",
    [
        (
            "I am flying from San Fransisco to Amsterdam",
            {
                "entity": ["O", "O", "O", "O", "city", "city", "O", "city"],
                "role": ["O", "O", "O", "O", "from", "from", "O", "to"],
            },
            {
                "entity": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
                "role": [1.0, 1.0, 1.0, 1.0, 0.98, 0.78, 1.0, 0.89],
            },
            [
                {
                    "entity": "city",
                    "start": 17,
                    "end": 30,
                    "value": "San Fransisco",
                    "role": "from",
                    "confidence_entity": 0.78,
                    "confidence_role": 0.78,
                },
                {
                    "entity": "city",
                    "start": 34,
                    "end": 43,
                    "value": "Amsterdam",
                    "role": "to",
                    "confidence_entity": 0.89,
                    "confidence_role": 0.89,
                },
            ],
        ),
        (
            "I am flying from San Fransisco to Amsterdam",
            {
                "entity": ["O", "O", "O", "O", "city", "city", "O", "city"],
                "group": ["O", "O", "O", "O", "1", "1", "O", "1"],
            },
            None,
            [
                {
                    "entity": "city",
                    "start": 17,
                    "end": 30,
                    "value": "San Fransisco",
                    "group": "1",
                },
                {
                    "entity": "city",
                    "start": 34,
                    "end": 43,
                    "value": "Amsterdam",
                    "group": "1",
                },
            ],
        ),
        (
            "Amsterdam",
            {"entity": ["city"], "role": ["O"], "group": ["O"]},
            None,
            [{"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"}],
        ),
        (
            "New-York",
            {"entity": ["city", "city"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 8, "value": "New-York"}],
        ),
        (
            "Amsterdam, Berlin, and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 11, "end": 17, "value": "Berlin"},
                {"entity": "city", "start": 23, "end": 29, "value": "London"},
            ],
        ),
        (
            "Amsterdam Berlin and London",
            {
                "entity": ["U-city", "U-city", "O", "U-city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 10, "end": 16, "value": "Berlin"},
                {"entity": "city", "start": 21, "end": 27, "value": "London"},
            ],
        ),
        (
            "San Fransisco Amsterdam, London",
            {
                "entity": ["B-city", "L-city", "U-city", "U-city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 13, "value": "San Fransisco"},
                {"entity": "city", "start": 14, "end": 23, "value": "Amsterdam"},
                {"entity": "city", "start": 25, "end": 31, "value": "London"},
            ],
        ),
        (
            "New York City Los Angeles and San Diego",
            {
                "entity": [
                    "B-city",
                    "I-city",
                    "L-city",
                    "B-city",
                    "L-city",
                    "O",
                    "B-city",
                    "L-city",
                ],
                "role": ["O", "O", "O", "O", "O", "O", "O", "O"],
                "group": ["O", "O", "O", "O", "O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 13, "value": "New York City"},
                {"entity": "city", "start": 14, "end": 25, "value": "Los Angeles"},
                {"entity": "city", "start": 30, "end": 39, "value": "San Diego"},
            ],
        ),
        (
            "Berlin weather",
            {"entity": ["I-city", "O"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 6, "value": "Berlin"}],
        ),
        (
            "New-York",
            {"entity": ["city", "city"], "role": ["O", "O"], "group": ["O", "O"]},
            None,
            [{"entity": "city", "start": 0, "end": 8, "value": "New-York"}],
        ),
        (
            "Amsterdam, Berlin, and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Amsterdam"},
                {"entity": "city", "start": 11, "end": 17, "value": "Berlin"},
                {"entity": "city", "start": 23, "end": 29, "value": "London"},
            ],
        ),
        (
            "Amsterdam Berlin and London",
            {
                "entity": ["city", "city", "O", "city"],
                "role": ["O", "O", "O", "O"],
                "group": ["O", "O", "O", "O"],
            },
            None,
            [
                {"entity": "city", "start": 0, "end": 16, "value": "Amsterdam Berlin"},
                {"entity": "city", "start": 21, "end": 27, "value": "London"},
            ],
        ),
    ],
)
def test_convert_tags_to_entities(
    text: Text,
    tags: Dict[Text, List[Text]],
    confidences: Dict[Text, List[float]],
    expected_entities: List[Dict[Text, Any]],
):
    extractor = EntityExtractor()
    tokenizer = WhitespaceTokenizer()

    message = Message(text)
    tokens = tokenizer.tokenize(message, TEXT)

    actual_entities = extractor.convert_predictions_into_entities(
        text, tokens, tags, confidences
    )
    assert actual_entities == expected_entities


@pytest.mark.parametrize(
    "text, warnings",
    [
        (
            "## intent:test\n"
            "- I want to fly from [Berlin](location) to [ San Fransisco](location)\n",
            1,
        ),
        (
            "## intent:test\n"
            "- I want to fly from [Berlin ](location) to [San Fransisco](location)\n",
            1,
        ),
        (
            "## intent:test\n"
            "- I want to fly from [Berlin](location) to [San Fransisco.](location)\n"
            "- I have nothing to say.",
            1,
        ),
        (
            "## intent:test\n"
            "- I have nothing to say.\n"
            "- I want to fly from [Berlin](location) to[San Fransisco](location)\n",
            1,
        ),
        (
            "## intent:test\n"
            "- I want to fly from [Berlin](location) to[San Fransisco](location)\n"
            "- Book a flight from [London](location) to [Paris.](location)\n",
            2,
        ),
    ],
)
def test_check_check_correct_entity_annotations(text: Text, warnings: int):
    reader = MarkdownReader()
    tokenizer = WhitespaceTokenizer()

    training_data = reader.reads(text)
    tokenizer.train(training_data)

    with pytest.warns(UserWarning) as record:
        EntityExtractor.check_correct_entity_annotations(training_data)

    assert len(record) == warnings
    assert all(
        [excerpt in record[0].message.args[0]]
        for excerpt in ["Misaligned entity annotation in sentence"]
    )
