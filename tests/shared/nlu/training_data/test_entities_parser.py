from typing import Text, List, Dict, Any

import pytest

import rasa.shared.nlu.training_data.entities_parser as entities_parser
from rasa.shared.exceptions import InvalidEntityFormatException, SchemaValidationError
from rasa.shared.nlu.constants import TEXT


@pytest.mark.parametrize(
    "example, expected_entities, expected_text",
    [
        (
            "I need an [economy class](travel_flight_class:economy) ticket from "
            '[boston]{"entity": "city", "role": "from"} to [new york]{"entity": "city",'
            ' "role": "to"}, please.',
            [
                {
                    "start": 10,
                    "end": 23,
                    "value": "economy",
                    "entity": "travel_flight_class",
                },
                {
                    "start": 36,
                    "end": 42,
                    "value": "boston",
                    "entity": "city",
                    "role": "from",
                },
                {
                    "start": 46,
                    "end": 54,
                    "value": "new york",
                    "entity": "city",
                    "role": "to",
                },
            ],
            "I need an economy class ticket from boston to new york, please.",
        ),
        ("i'm looking for a place to eat", [], "i'm looking for a place to eat"),
        (
            "i'm looking for a place in the [north](loc-direction) of town",
            [{"start": 31, "end": 36, "value": "north", "entity": "loc-direction"}],
            "i'm looking for a place in the north of town",
        ),
        (
            "show me [chines](cuisine:chinese) restaurants",
            [{"start": 8, "end": 14, "value": "chinese", "entity": "cuisine"}],
            "show me chines restaurants",
        ),
        (
            'show me [italian]{"entity": "cuisine", "value": "22_ab-34*3.A:43er*+?df"} '
            "restaurants",
            [
                {
                    "start": 8,
                    "end": 15,
                    "value": "22_ab-34*3.A:43er*+?df",
                    "entity": "cuisine",
                }
            ],
            "show me italian restaurants",
        ),
        ("Do you know {ABC} club?", [], "Do you know {ABC} club?"),
        (
            "show me [chines](22_ab-34*3.A:43er*+?df) restaurants",
            [{"start": 8, "end": 14, "value": "43er*+?df", "entity": "22_ab-34*3.A"}],
            "show me chines restaurants",
        ),
        (
            'I want to fly from [Berlin]{"entity": "city", "role": "to"} to [LA]{'
            '"entity": "city", "role": "from", "value": "Los Angeles"}',
            [
                {
                    "start": 19,
                    "end": 25,
                    "value": "Berlin",
                    "entity": "city",
                    "role": "to",
                },
                {
                    "start": 29,
                    "end": 31,
                    "value": "Los Angeles",
                    "entity": "city",
                    "role": "from",
                },
            ],
            "I want to fly from Berlin to LA",
        ),
        (
            'I want to fly from [Berlin](city) to [LA]{"entity": "city", "role": '
            '"from", "value": "Los Angeles"}',
            [
                {"start": 19, "end": 25, "value": "Berlin", "entity": "city"},
                {
                    "start": 29,
                    "end": 31,
                    "value": "Los Angeles",
                    "entity": "city",
                    "role": "from",
                },
            ],
            "I want to fly from Berlin to LA",
        ),
        (
            'I want to fly from [Berlin](city) to [LA][{"entity": "city", "role": '
            '"from", "value": "Los Angeles"}, {"entity": "location", "value": '
            '"Los Angeles"}]',
            [
                {"start": 19, "end": 25, "value": "Berlin", "entity": "city"},
                {
                    "start": 29,
                    "end": 31,
                    "value": "Los Angeles",
                    "entity": "city",
                    "role": "from",
                },
                {
                    "start": 29,
                    "end": 31,
                    "value": "Los Angeles",
                    "entity": "location",
                },
            ],
            "I want to fly from Berlin to LA",
        ),
    ],
)
def test_markdown_entity_regex(
    example: Text, expected_entities: List[Dict[Text, Any]], expected_text: Text
):

    result = entities_parser.find_entities_in_training_example(example)
    assert result == expected_entities

    replaced_text = entities_parser.replace_entities(example)
    assert replaced_text == expected_text


def test_parse_training_example():
    message = entities_parser.parse_training_example("Hello!", intent="greet")
    assert message.get("intent") == "greet"
    assert message.get(TEXT) == "Hello!"


def test_parse_empty_example():
    message = entities_parser.parse_training_example("")
    assert message.get("intent") is None
    assert message.get(TEXT) == ""


def test_parse_training_example_with_entities():
    message = entities_parser.parse_training_example(
        "I am from [Berlin](city).", intent="inform"
    )
    assert message.get("intent") == "inform"
    assert message.get(TEXT) == "I am from Berlin."
    assert message.get("entities") == [
        {"start": 10, "end": 16, "value": "Berlin", "entity": "city"}
    ]


def test_markdown_entity_regex_error_handling_not_json():
    with pytest.raises(InvalidEntityFormatException):
        entities_parser.find_entities_in_training_example(
            # JSON syntax error: missing closing " for `role`
            'I want to fly from [Berlin]{"entity": "city", "role: "from"}'
        )


def test_markdown_entity_regex_error_handling_wrong_schema():
    with pytest.raises(SchemaValidationError):
        entities_parser.find_entities_in_training_example(
            # Schema error: "entiti" instead of "entity"
            'I want to fly from [Berlin]{"entiti": "city", "role": "from"}'
        )
