from typing import Any, Text, Dict, List

import pytest

from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import ENTITIES
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.lookup_entity_extractor import LookupEntityExtractor


@pytest.mark.parametrize(
    "text, lookup, expected_entities",
    [
        (
            "Berlin and London are cities.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 0,
                    "end": 6,
                    "extractor": "LookupEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "LookupEntityExtractor",
                },
            ],
        ),
        (
            "Sophie is visiting Thomas in Berlin.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 29,
                    "end": 35,
                    "extractor": "LookupEntityExtractor",
                },
                {
                    "entity": "person",
                    "value": "Sophie",
                    "start": 0,
                    "end": 6,
                    "extractor": "LookupEntityExtractor",
                },
            ],
        ),
        (
            "Rasa is great.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [],
        ),
    ],
)
def test_process(
    text: Text,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(text)

    training_data = TrainingData()
    training_data.lookup_tables = lookup

    entity_extractor = LookupEntityExtractor()
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


@pytest.mark.parametrize(
    "text, lowercase, lookup, expected_entities",
    [
        (
            "berlin and London are cities.",
            False,
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "LookupEntityExtractor",
                }
            ],
        ),
        (
            "berlin and London are cities.",
            True,
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "london"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "berlin",
                    "start": 0,
                    "end": 6,
                    "extractor": "LookupEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "LookupEntityExtractor",
                },
            ],
        ),
    ],
)
def test_lowercase(
    text: Text,
    lowercase: bool,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(text)
    training_data = TrainingData()
    training_data.lookup_tables = lookup

    entity_extractor = LookupEntityExtractor({"lowercase": lowercase})
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


def test_do_not_overwrite_any_entities():
    message = Message("Max lives in Berlin.")
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    training_data = TrainingData()
    training_data.lookup_tables = [
        {"name": "city", "elements": ["London", "Berlin", "Amsterdam"]}
    ]

    entity_extractor = LookupEntityExtractor()
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == [
        {"entity": "person", "value": "Max", "start": 0, "end": 3},
        {
            "entity": "city",
            "value": "Berlin",
            "start": 13,
            "end": 19,
            "extractor": "LookupEntityExtractor",
        },
    ]
