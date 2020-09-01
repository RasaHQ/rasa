import textwrap
from typing import Text

import pytest

import rasa.utils.io as io_utils
from rasa.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.core.domain import InvalidDomain
from rasa.nlu.constants import INTENT
from rasa.nlu.training_data.formats.rasa_yaml import RasaYAMLReader, RasaYAMLWriter

MULTILINE_INTENT_EXAMPLES = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- intent: intent_name
  examples: |
    - how much CO2 will that use?
    - how much carbon will a one way flight from [new york]{{"entity": "city", "role": "from"}} to california produce?
"""

MULTILINE_INTENT_EXAMPLE_WITH_SYNONYM = """
nlu:
- intent: intent_name
  examples: |
    - flight from [boston]{"entity": "city", "role": "from", "value": "bostn"}?
"""

MULTILINE_INTENT_EXAMPLES_NO_LEADING_SYMBOL = """
nlu:
- intent: intent_name
  examples: |
    how much CO2 will that use?
    - how much carbon will a one way flight from [new york]{"entity": "city", "role": "from"} to california produce?
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

MINIMAL_VALID_EXAMPLE = """
nlu:\n
stories:
"""

WRONG_YAML_NLU_CONTENT_1 = """
nlu:
- intent: name
  non_key: value
"""

WRONG_YAML_NLU_CONTENT_2 = """
nlu:
- intent: greet
  examples: |
  - Hi
  - Hey
"""

SYNONYM_EXAMPLE = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- synonym: savings
  examples: |
    - pink pig
    - savings account
"""

LOOKUP_ITEM_NAME = "additional_currencies"
LOOKUP_EXAMPLE = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- lookup: {LOOKUP_ITEM_NAME}
  examples: |
    - Peso
    - Euro
    - Dollar
"""

REGEX_NAME = "zipcode"
PATTERN_1 = "[0-9]{4}"
PATTERN_2 = "[0-9]{5}"
REGEX_EXAMPLE = f"""
version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
nlu:
- regex: {REGEX_NAME}
  examples: |
    - {PATTERN_1}
    - {PATTERN_2}
"""


def test_wrong_format_raises():

    wrong_yaml_nlu_content = """
    !!
    """

    parser = RasaYAMLReader()
    with pytest.raises(ValueError):
        parser.reads(wrong_yaml_nlu_content)


@pytest.mark.parametrize(
    "example", [WRONG_YAML_NLU_CONTENT_1, WRONG_YAML_NLU_CONTENT_2]
)
def test_wrong_schema_raises(example: Text):

    parser = RasaYAMLReader()
    with pytest.raises(ValueError):
        parser.reads(example)


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
    assert not len(training_data.entity_synonyms)


# This test would work only with examples that have a `version` key specified
@pytest.mark.parametrize(
    "example",
    [MULTILINE_INTENT_EXAMPLES, SYNONYM_EXAMPLE, LOOKUP_EXAMPLE, REGEX_EXAMPLE],
)
def test_yaml_examples_are_written(example: Text):
    parser = RasaYAMLReader()
    writer = RasaYAMLWriter()

    training_data = parser.reads(example)
    assert example.strip() == writer.dumps(training_data).strip()


def test_multiline_intent_example_is_skipped_when_no_leading_symbol():
    parser = RasaYAMLReader()

    with pytest.warns(None) as record:
        training_data = parser.reads(MULTILINE_INTENT_EXAMPLES_NO_LEADING_SYMBOL)

    # warning for the missing leading symbol
    assert len(record) == 1

    assert len(training_data.training_examples) == 1
    assert not len(training_data.entity_synonyms)


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
    - {example}
"""

    result = reader.reads(yaml_string)

    assert len(result.training_examples) == 1
    actual_example = result.training_examples[0]
    assert actual_example.data["intent"] == intent_name
    assert len(actual_example.data.get("entities", [])) == expected_num_entities


def test_synonyms_are_parsed():
    parser = RasaYAMLReader()
    training_data = parser.reads(SYNONYM_EXAMPLE)

    assert len(training_data.entity_synonyms) == 2
    assert training_data.entity_synonyms["pink pig"] == "savings"
    assert training_data.entity_synonyms["savings account"] == "savings"


def test_synonyms_are_extracted_from_entities():
    parser = RasaYAMLReader()
    training_data = parser.reads(MULTILINE_INTENT_EXAMPLE_WITH_SYNONYM)

    assert len(training_data.entity_synonyms) == 1


def test_lookup_is_parsed():

    parser = RasaYAMLReader()
    training_data = parser.reads(LOOKUP_EXAMPLE)

    assert training_data.lookup_tables[0]["name"] == LOOKUP_ITEM_NAME
    assert len(training_data.lookup_tables[0]["elements"]) == 3


def test_regex_is_parsed():

    parser = RasaYAMLReader()
    training_data = parser.reads(REGEX_EXAMPLE)

    assert len(training_data.regex_features) == 2
    assert {"name": REGEX_NAME, "pattern": PATTERN_1} in training_data.regex_features
    assert {"name": REGEX_NAME, "pattern": PATTERN_2} in training_data.regex_features


def test_minimal_valid_example():
    parser = RasaYAMLReader()

    with pytest.warns(None) as record:
        parser.reads(MINIMAL_VALID_EXAMPLE)

    assert not len(record)


def test_minimal_yaml_nlu_file(tmp_path):
    target_file = tmp_path / "test_nlu_file.yaml"
    io_utils.write_yaml(MINIMAL_VALID_EXAMPLE, target_file, True)
    assert RasaYAMLReader.is_yaml_nlu_file(target_file)


def test_nlg_reads_text():
    responses_yml = textwrap.dedent(
        """
      responses:
        chitchat/ask_weather:
        - text: Where do you want to check the weather?
    """
    )

    reader = RasaYAMLReader()
    result = reader.reads(responses_yml)

    assert result.responses == {
        "chitchat/ask_weather": [{"text": "Where do you want to check the weather?"}]
    }


def test_nlg_reads_any_multimedia():
    responses_yml = textwrap.dedent(
        """
      responses:
        chitchat/ask_weather:
        - text: Where do you want to check the weather?
          image: https://example.com/weather.jpg
    """
    )

    reader = RasaYAMLReader()
    result = reader.reads(responses_yml)

    assert result.responses == {
        "chitchat/ask_weather": [
            {
                "text": "Where do you want to check the weather?",
                "image": "https://example.com/weather.jpg",
            }
        ]
    }


def test_nlg_fails_to_read_empty():
    responses_yml = textwrap.dedent(
        """
      responses:
    """
    )

    reader = RasaYAMLReader()

    with pytest.raises(ValueError):
        reader.reads(responses_yml)


def test_nlg_fails_on_empty_response():
    responses_yml = textwrap.dedent(
        """
      responses:
        chitchat/ask_weather:
    """
    )

    reader = RasaYAMLReader()

    with pytest.raises(ValueError):
        reader.reads(responses_yml)


def test_nlg_multimedia_load_dump_roundtrip():
    responses_yml = textwrap.dedent(
        """
      responses:
        chitchat/ask_weather:
        - text: Where do you want to check the weather?
          image: https://example.com/weather.jpg

        chitchat/ask_name:
        - text: My name is Sara.
    """
    )

    reader = RasaYAMLReader()
    result = reader.reads(responses_yml)

    dumped = RasaYAMLWriter().dumps(result)

    validation_reader = RasaYAMLReader()
    dumped_result = validation_reader.reads(dumped)

    assert dumped_result.responses == result.responses

    # dumping again should also not change the format
    assert dumped == RasaYAMLWriter().dumps(dumped_result)
