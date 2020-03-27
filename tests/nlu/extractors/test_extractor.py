from typing import Any, Text, Dict, List

import pytest

from nlu.tokenizers.tokenizer import Token
from nlu.training_data import Message
from rasa.nlu.extractors.extractor import EntityExtractor


@pytest.mark.parametrize(
    "text, tokens, entities, keep, expected_entities",
    [
        (
            "Aarhus is a city",
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
            "Aarhus",
            [Token("Aar", 0, 3), Token("hus", 3, 6)],
            [
                {"entity": "iata", "start": 0, "end": 3, "value": "Aar"},
                {"entity": "city", "start": 3, "end": 6, "value": "hus"},
            ],
            True,
            [],
        ),
        (
            "Aarhus city",
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
            "Aarhus is a city",
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
            "Aarhus",
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
            "Aarhus",
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
            "Aarhus",
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
        (
            "Buenos Aires is a city",
            [
                Token("Buenos", 0, 6),
                Token("Ai", 7, 9),
                Token("res", 9, 12),
                Token("is", 13, 15),
                Token("a", 16, 17),
                Token("city", 18, 22),
            ],
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Buenos Ai"},
                {"entity": "location", "start": 18, "end": 22, "value": "city"},
            ],
            False,
            [{"entity": "location", "start": 18, "end": 22, "value": "city"}],
        ),
        (
            "Buenos Aires is a city",
            [
                Token("Buenos", 0, 6),
                Token("Ai", 7, 9),
                Token("res", 9, 12),
                Token("is", 13, 15),
                Token("a", 16, 17),
                Token("city", 18, 22),
            ],
            [
                {"entity": "city", "start": 0, "end": 9, "value": "Buenos Ai"},
                {"entity": "location", "start": 18, "end": 22, "value": "city"},
            ],
            True,
            [
                {"entity": "city", "start": 0, "end": 12, "value": "Buenos Aires"},
                {"entity": "location", "start": 18, "end": 22, "value": "city"},
            ],
        ),
        (
            "Buenos Aires is a city",
            [
                Token("Buen", 0, 4),
                Token("os", 4, 6),
                Token("Ai", 7, 9),
                Token("res", 9, 12),
                Token("is", 13, 15),
                Token("a", 16, 17),
                Token("city", 18, 22),
            ],
            [
                {"entity": "city", "start": 4, "end": 9, "value": "os Ai"},
                {"entity": "location", "start": 18, "end": 22, "value": "city"},
            ],
            True,
            [
                {"entity": "city", "start": 0, "end": 12, "value": "Buenos Aires"},
                {"entity": "location", "start": 18, "end": 22, "value": "city"},
            ],
        ),
    ],
)
def test_convert_tags_to_entities(
    text: Text,
    tokens: List[Token],
    entities: List[Dict[Text, Any]],
    keep: bool,
    expected_entities: List[Dict[Text, Any]],
):
    extractor = EntityExtractor()

    message = Message(text)
    message.set("tokens", tokens)

    updated_entities = extractor.clean_up_entities(message, entities, keep)

    assert updated_entities == expected_entities
