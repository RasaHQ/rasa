from typing import Any, Text, Dict, List

import pytest

from rasa.nlu.extractors.extractor import EntityExtractor


@pytest.mark.parametrize(
    "entities, keep, expected_entities",
    [
        (
            [
                {
                    "entity": "iata",
                    "start": 0,
                    "end": 3,
                    "extractor": "DIETClassifier",
                    "value": "Aar",
                },
                {
                    "entity": "city",
                    "start": 3,
                    "end": 6,
                    "extractor": "DIETClassifier",
                    "value": "hus",
                },
            ],
            False,
            [],
        ),
        (
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 3,
                    "extractor": "DIETClassifier",
                    "value": "Aarhus",
                },
                {
                    "entity": "type",
                    "start": 4,
                    "end": 9,
                    "extractor": "DIETClassifier",
                    "value": "city",
                },
            ],
            False,
            [
                {
                    "entity": "city",
                    "start": 0,
                    "end": 3,
                    "extractor": "DIETClassifier",
                    "value": "Aarhus",
                },
                {
                    "entity": "type",
                    "start": 4,
                    "end": 9,
                    "extractor": "DIETClassifier",
                    "value": "city",
                },
            ],
        ),
        (
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
                    "end": 5,
                    "confidence": 0.43,
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
    ],
)
def test_convert_tags_to_entities(
    entities: List[Dict[Text, Any]],
    keep: bool,
    expected_entities: List[Dict[Text, Any]],
):
    extractor = EntityExtractor()

    updated_entities = extractor.clean_up_entities(entities, keep)

    assert updated_entities == expected_entities
