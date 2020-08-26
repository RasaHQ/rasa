from typing import Optional, Text, Dict, Any, List

import pytest

from rasa.nlu import load_data
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.training_data.formats import RasaReader
from rasa.nlu.training_data.formats import MarkdownReader, MarkdownWriter


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
        (
            'I want to travel to [Cairo]{"entity": "city"} \\n'
            'Also, do you have flights to [Alexandria]{"entity": "city"}?',
            2,
        ),
    ],
)
def test_markdown_entity_regex(example: Text, expected_num_entities: int):
    r = MarkdownReader()

    md = f"""
## intent:test-intent
- {example}
    """

    result = r.reads(md)

    assert len(result.training_examples) == 1
    actual_example = result.training_examples[0]
    assert actual_example.data["intent"] == "test-intent"
    assert len(actual_example.data.get("entities", [])) == expected_num_entities


def test_markdown_empty_section():
    data = load_data("data/test/markdown_single_sections/empty_section.md")
    assert data.regex_features == [{"name": "greet", "pattern": r"hey[^\s]*"}]

    assert not data.entity_synonyms
    assert len(data.lookup_tables) == 1
    assert data.lookup_tables[0]["name"] == "chinese"
    assert "Chinese" in data.lookup_tables[0]["elements"]
    assert "Chines" in data.lookup_tables[0]["elements"]


def test_markdown_not_existing_section():
    with pytest.raises(ValueError):
        load_data("data/test/markdown_single_sections/not_existing_section.md")


def test_section_value_with_delimiter():
    td_section_with_delimiter = load_data(
        "data/test/markdown_single_sections/section_with_delimiter.md"
    )
    assert td_section_with_delimiter.entity_synonyms == {"10:00 am": "10:00"}


def test_markdown_order():
    r = MarkdownReader()

    md = """## intent:z
- i'm looking for a place to eat
- i'm looking for a place in the [north](loc-direction) of town

## intent:a
- intent a
- also very important
"""

    training_data = r.reads(md)
    assert training_data.nlu_as_markdown() == md


def test_markdown_unespace_tokens():
    r = MarkdownReader()

    md = """## intent:test-intent
- Hi \\t Can you help me?\\n I want to go to [Alexandria]{"entity": "city"}
"""
    expected_num_entities = 1

    training_data = r.reads(md)
    assert len(training_data.training_examples) == 1

    actual_example = training_data.training_examples[0]
    assert actual_example.data["intent"] == "test-intent"
    assert len(actual_example.data.get("entities", [])) == expected_num_entities


def test_dump_nlu_with_responses():
    md = """## intent:greet
- hey
- howdy
- hey there
- hello
- hi
- good morning
- good evening
- dear sir

## intent:chitchat/ask_name
- What's your name?
- What can I call you?

## intent:chitchat/ask_weather
- How's the weather?
- Is it too hot outside?
"""

    r = MarkdownReader()
    nlu_data = r.reads(md)

    dumped = nlu_data.nlu_as_markdown()
    assert dumped == md


@pytest.mark.parametrize(
    "entity_extractor,expected_output",
    [
        (None, '- [test]{"entity": "word", "value": "random"}'),
        ("", '- [test]{"entity": "word", "value": "random"}'),
        ("random-extractor", '- [test]{"entity": "word", "value": "random"}'),
        (CRFEntityExtractor.__name__, '- [test]{"entity": "word", "value": "random"}'),
        (DucklingHTTPExtractor.__name__, "- test"),
        (SpacyEntityExtractor.__name__, "- test"),
        (
            MitieEntityExtractor.__name__,
            '- [test]{"entity": "word", "value": "random"}',
        ),
    ],
)
def test_dump_trainable_entities(
    entity_extractor: Optional[Text], expected_output: Text
):
    training_data_json = {
        "rasa_nlu_data": {
            "common_examples": [
                {
                    "text": "test",
                    "intent": "greet",
                    "entities": [
                        {"start": 0, "end": 4, "value": "random", "entity": "word"}
                    ],
                }
            ]
        }
    }
    if entity_extractor is not None:
        training_data_json["rasa_nlu_data"]["common_examples"][0]["entities"][0][
            "extractor"
        ] = entity_extractor

    training_data_object = RasaReader().read_from_json(training_data_json)
    md_dump = MarkdownWriter().dumps(training_data_object)
    assert md_dump.splitlines()[1] == expected_output


@pytest.mark.parametrize(
    "entity, expected_output",
    [
        (
            {
                "start": 0,
                "end": 4,
                "value": "random",
                "entity": "word",
                "role": "role-name",
                "group": "group-name",
            },
            '- [test]{"entity": "word", "role": "role-name", "group": "group-name", '
            '"value": "random"}',
        ),
        ({"start": 0, "end": 4, "entity": "word"}, "- [test](word)"),
        (
            {
                "start": 0,
                "end": 4,
                "entity": "word",
                "role": "role-name",
                "group": "group-name",
            },
            '- [test]{"entity": "word", "role": "role-name", "group": "group-name"}',
        ),
        (
            {"start": 0, "end": 4, "entity": "word", "value": "random"},
            '- [test]{"entity": "word", "value": "random"}',
        ),
    ],
)
def test_dump_entities(entity: Dict[Text, Any], expected_output: Text):
    training_data_json = {
        "rasa_nlu_data": {
            "common_examples": [
                {"text": "test", "intent": "greet", "entities": [entity]}
            ]
        }
    }
    training_data_object = RasaReader().read_from_json(training_data_json)
    md_dump = MarkdownWriter().dumps(training_data_object)
    assert md_dump.splitlines()[1] == expected_output
