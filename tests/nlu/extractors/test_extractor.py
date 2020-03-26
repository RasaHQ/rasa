from typing import Any, Text, Dict, List

import pytest

from nlu.tokenizers.tokenizer import Token
from nlu.training_data import Message
from rasa.nlu.extractors.extractor import EntityExtractor


@pytest.mark.parametrize(
    "tokens, entities, keep, expected_entities",
    [
        (
            [
                Token("Aar", 0, 3),
                Token("hus", 3, 6),
                Token("is", 7, 9),
                Token("a", 10, 11),
                Token("city", 12, 16),
            ],
            [
                {"entity": "iata", "start": 0, "end": 3, "value": "Aar"},
                {"entity": "city", "start": 3, "end": 6, "value": "hus"},
                {"entity": "location", "start": 12, "end": 16, "value": "city"},
            ],
            False,
            [{"entity": "location", "start": 12, "end": 16, "value": "city"}],
        ),
        (
            [Token("Aar", 0, 3), Token("hus", 3, 6)],
            [
                {"entity": "iata", "start": 0, "end": 3, "value": "Aar"},
                {"entity": "city", "start": 3, "end": 6, "value": "hus"},
            ],
            True,
            [],
        ),
        (
            [Token("Aarhus", 0, 6), Token("city", 7, 11)],
            [
                {"entity": "city", "start": 0, "end": 6, "value": "Aarhus"},
                {"entity": "type", "start": 7, "end": 11, "value": "city"},
            ],
            False,
            [
                {"entity": "city", "start": 0, "end": 6, "value": "Aarhus"},
                {"entity": "type", "start": 7, "end": 11, "value": "city"},
            ],
        ),
        (
            [
                Token("Aar", 0, 3),
                Token("hus", 3, 6),
                Token("is", 7, 9),
                Token("a", 10, 11),
                Token("city", 12, 16),
            ],
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 3,
                    "confidence": 0.87,
                    "value": "Aar",
                },
                {
                    "entity": "iata",
                    "start": 3,
                    "end": 6,
                    "confidence": 0.43,
                    "value": "hus",
                },
                {"entity": "location", "start": 12, "end": 16, "value": "city"},
            ],
            True,
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 6,
                    "confidence": 0.87,
                    "value": "Aarhus",
                },
                {"entity": "location", "start": 12, "end": 16, "value": "city"},
            ],
        ),
        (
            [Token("Aa", 0, 2), Token("r", 2, 3), Token("hu", 3, 5), Token("s", 5, 6)],
            [
                {
                    "entity": "iata",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.32,
                    "value": "Aa",
                },
                {
                    "entity": "city",
                    "start": 2,
                    "end": 3,
                    "confidence": 0.87,
                    "value": "r",
                },
                {
                    "entity": "iata",
                    "start": 3,
                    "end": 5,
                    "confidence": 0.21,
                    "value": "hu",
                },
                {
                    "entity": "city",
                    "start": 5,
                    "end": 6,
                    "confidence": 0.43,
                    "value": "s",
                },
            ],
            True,
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 6,
                    "confidence": 0.87,
                    "value": "Aarhus",
                }
            ],
        ),
        (
            [Token("Aa", 0, 2), Token("r", 2, 3), Token("hu", 3, 5), Token("s", 5, 6)],
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.32,
                    "value": "Aa",
                }
            ],
            True,
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 6,
                    "confidence": 0.32,
                    "value": "Aarhus",
                }
            ],
        ),
        (
            [Token("Aa", 0, 2), Token("r", 2, 3), Token("hu", 3, 5), Token("s", 5, 6)],
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.32,
                    "value": "Aa",
                }
            ],
            False,
            [],
        ),
    ],
)
def test_convert_tags_to_entities(
    tokens: List[Token],
    entities: List[Dict[Text, Any]],
    keep: bool,
    expected_entities: List[Dict[Text, Any]],
):
    extractor = EntityExtractor()

    message = Message("test")
    message.set("tokens", tokens)

    updated_entities = extractor.clean_up_entities(message, entities, keep)

    assert updated_entities == expected_entities
