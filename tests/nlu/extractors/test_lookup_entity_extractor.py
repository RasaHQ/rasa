from typing import Any, Text, Dict, List

import pytest

from rasa.nlu.constants import ENTITIES
from rasa.nlu.training_data import Message
from rasa.nlu.extractors.lookup_entity_extractor import LookupEntityExtractor
from tests.nlu.utilities import write_tmp_file


@pytest.mark.parametrize(
    "component_config",
    [
        {},
        {"lookup": None},
        {"lookup": {}},
        {
            "lookup": {
                "some-entity": "some/invalid/path.txt",
                "some-other-entity": "some/other/invalid/path.txt",
            }
        },
    ],
)
def test_raise_error_on_invalid_config(component_config: Dict[Text, Any]):
    with pytest.raises(ValueError):
        LookupEntityExtractor(component_config)


@pytest.mark.parametrize(
    "text, lookup, expected_entities",
    [
        (
            "Berlin and London are cities.",
            {"city": ["Berlin", "Amsterdam", "New York", "London"]},
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
            {
                "city": ["Berlin", "Amsterdam", "New York", "London"],
                "person": ["Max", "John", "Sophie", "Lisa"],
            },
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
            {
                "city": ["Berlin", "Amsterdam", "New York", "London"],
                "person": ["Max", "John", "Sophie", "Lisa"],
            },
            [],
        ),
    ],
)
def test_process(
    text: Text, lookup: Dict[Text, List[Text]], expected_entities: List[Dict[Text, Any]]
):
    message = Message(text)

    lookup_table = {}
    for entity, examples in lookup.items():
        file_path = write_tmp_file(examples)
        lookup_table[entity] = file_path

    entity_extractor = LookupEntityExtractor({"lookup": lookup_table})
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


@pytest.mark.parametrize(
    "text, lowercase, lookup, expected_entities",
    [
        (
            "berlin and London are cities.",
            False,
            {"city": ["Berlin", "Amsterdam", "New York", "London"]},
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
            {"city": ["Berlin", "Amsterdam", "New York", "london"]},
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
    lookup: Dict[Text, List[Text]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(text)

    lookup_table = {}
    for entity, examples in lookup.items():
        file_path = write_tmp_file(examples)
        lookup_table[entity] = file_path

    entity_extractor = LookupEntityExtractor(
        {"lookup": lookup_table, "lowercase": lowercase}
    )
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


def test_do_not_overwrite_any_entities():
    message = Message("Max lives in Berlin.")
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    lookup_table = {}
    file_path = write_tmp_file(["London", "Berlin", "Amsterdam"])
    lookup_table["city"] = file_path

    entity_extractor = LookupEntityExtractor({"lookup": lookup_table})
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
