from typing import Any, Text, Dict, List

import pytest

from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import ENTITIES, TEXT, INTENT
from rasa.nlu.extractors.regex_entity_extractor import RegexEntityExtractor


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
                    "extractor": "RegexEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "RegexEntityExtractor",
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
                    "extractor": "RegexEntityExtractor",
                },
                {
                    "entity": "person",
                    "value": "Sophie",
                    "start": 0,
                    "end": 6,
                    "extractor": "RegexEntityExtractor",
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
    message = Message(data={TEXT: text})

    training_data = TrainingData()
    training_data.lookup_tables = lookup
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [{"entity": "person", "value": "Max"}],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]

    entity_extractor = RegexEntityExtractor()
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


@pytest.mark.parametrize(
    "text, case_sensitive, lookup, expected_entities",
    [
        (
            "berlin and London are cities.",
            True,
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
                    "extractor": "RegexEntityExtractor",
                }
            ],
        ),
        (
            "berlin and London are cities.",
            False,
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
                    "extractor": "RegexEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "RegexEntityExtractor",
                },
            ],
        ),
    ],
)
def test_lowercase(
    text: Text,
    case_sensitive: bool,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(data={TEXT: text})
    training_data = TrainingData()
    training_data.lookup_tables = lookup
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [{"entity": "person", "value": "Max"}],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]

    entity_extractor = RegexEntityExtractor({"case_sensitive": case_sensitive})
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


def test_do_not_overwrite_any_entities():
    message = Message(data={TEXT: "Max lives in Berlin.", INTENT: "infrom"})
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    training_data = TrainingData()
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [{"entity": "person", "value": "Max"}],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]
    training_data.lookup_tables = [
        {"name": "city", "elements": ["London", "Berlin", "Amsterdam"]}
    ]

    entity_extractor = RegexEntityExtractor()
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
            "extractor": "RegexEntityExtractor",
        },
    ]
