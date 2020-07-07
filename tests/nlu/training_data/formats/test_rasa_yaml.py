from typing import Text

import pytest
from ruamel.yaml import YAMLError

from rasa.nlu.constants import INTENT
from rasa.nlu.training_data.formats.rasa_yaml import RasaYAMLReader

MULTILINE_INTENT_EXAMPLES = """
nlu:
- intent: intent_name
  examples: |
     how much CO2 will that use?
     how much carbon will a one way flight from [new york]{"entity": "city", "role": "from"} to california produce?
"""

INTENT_EXAMPLES_WITH_METADATA = """
nlu:
- intent: intent_name
  metadata:
  examples:
  - text: |
      how much CO2 will that use?
    metadata:
      sentiment: positive
  - text: |
      how much carbon will a one way flight from [new york]{"entity": "city", "role": "from"} to california produce?
"""


def test_wrong_format_raises():

    wrong_yaml_nlu_content = """
    !!
    """

    parser = RasaYAMLReader()

    with pytest.raises(YAMLError):
        parser.reads(wrong_yaml_nlu_content)


@pytest.mark.parametrize(
    "example", [MULTILINE_INTENT_EXAMPLES, INTENT_EXAMPLES_WITH_METADATA]
)
def test_multiline_intent_is_parsed(example: Text):
    parser = RasaYAMLReader()

    with pytest.warns(None) as record:
        training_data = parser.reads(example)

    assert not len(record)

    assert len(training_data.training_examples) == 2
    assert training_data.training_examples[0].get(
        INTENT
    ) == training_data.training_examples[1].get(INTENT)


@pytest.mark.parametrize(
    "example, expected_num_entities",
    [
        (
            "I need an [economy class](travel_flight_class:economy) ticket from "
            '[boston]{"entity": "city", "role": "from"} to [new york]{"entity": "city",'
            ' "role": "to"}, please.',
            3,
        ),
        ("i'm looking for a place to eat", 0),
        ("i'm looking for a place in the [north](loc-direction) of town", 1),
        ("show me [chines](cuisine:chinese) restaurants", 1),
        (
            'show me [italian]{"entity": "cuisine", "value": "22_ab-34*3.A:43er*+?df"} '
            "restaurants",
            1,
        ),
        ("Do you know {ABC} club?", 0),
        ("show me [chines](22_ab-34*3.A:43er*+?df) restaurants", 1),
        (
            'I want to fly from [Berlin]{"entity": "city", "role": "to"} to [LA]{'
            '"entity": "city", "role": "from", "value": "Los Angeles"}',
            2,
        ),
        (
            'I want to fly from [Berlin](city) to [LA]{"entity": "city", "role": '
            '"from", "value": "Los Angeles"}',
            2,
        ),
    ],
)
def test_entity_is_extracted(example: Text, expected_num_entities: int):
    reader = RasaYAMLReader()

    intent_name = "test-intent"

    yaml_string = f"""
nlu:
- intent: {intent_name}
  examples: |
    {example}
"""

    result = reader.reads(yaml_string)

    assert len(result.training_examples) == 1
    actual_example = result.training_examples[0]
    assert actual_example.data["intent"] == intent_name
    assert len(actual_example.data.get("entities", [])) == expected_num_entities


def test_synonyms_are_parsed():
    synonym_example = """
    nlu:
    - synonym: savings
      examples: |
        pink pig
        savings account
    """

    parser = RasaYAMLReader()
    training_data = parser.reads(synonym_example)

    assert len(training_data.entity_synonyms) == 2
    assert training_data.entity_synonyms["pink pig"] == "savings"
    assert training_data.entity_synonyms["savings account"] == "savings"


def test_lookup_is_parsed():

    lookup_item_name = "additional_currencies"

    lookup_example = f"""
    nlu:
    - lookup: {lookup_item_name}
      examples: |
        Peso
        Euro
        Dollar
    """

    parser = RasaYAMLReader()
    training_data = parser.reads(lookup_example)

    assert training_data.lookup_tables[0]["name"] == lookup_item_name
    assert len(training_data.lookup_tables[0]["elements"]) == 3


def test_regex_is_parsed():

    regex_name = "zipcode"
    pattern_1 = "[0-9]{5}"
    pattern_2 = "[0-9]{4}"

    regex_example = f"""
    nlu:
    - regex: {regex_name}
      examples: |
        {pattern_1}
        {pattern_2}
    """

    parser = RasaYAMLReader()
    training_data = parser.reads(regex_example)

    assert len(training_data.regex_features) == 2
    assert {"name": regex_name, "pattern": pattern_1} in training_data.regex_features
    assert {"name": regex_name, "pattern": pattern_2} in training_data.regex_features
