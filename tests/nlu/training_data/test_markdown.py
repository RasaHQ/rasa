from typing import Optional, Text, Dict, Any

import pytest

from nlu import load_data
from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor
from rasa.nlu.extractors.duckling_http_extractor import DucklingHTTPExtractor
from rasa.nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa.nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa.nlu.training_data.formats import RasaReader
from rasa.nlu.training_data.formats import MarkdownReader, MarkdownWriter


def test_markdown_entity_regex():
    r = MarkdownReader()

    md = """
## intent:restaurant_search
- i'm looking for a place to eat
- i'm looking for a place in the [north](loc-direction) of town
- show me [chines](cuisine:chinese) restaurants
- show me [chines](22_ab-34*3.A:43er*+?df) restaurants
    """

    result = r.reads(md)

    assert len(result.training_examples) == 4
    first = result.training_examples[0]
    assert first.data == {"intent": "restaurant_search"}
    assert first.text == "i'm looking for a place to eat"

    second = result.training_examples[1]
    assert second.data == {
        "intent": "restaurant_search",
        "entities": [
            {"start": 31, "end": 36, "value": "north", "entity": "loc-direction"}
        ],
    }
    assert second.text == "i'm looking for a place in the north of town"

    third = result.training_examples[2]
    assert third.data == {
        "intent": "restaurant_search",
        "entities": [{"start": 8, "end": 14, "value": "chinese", "entity": "cuisine"}],
    }
    assert third.text == "show me chines restaurants"

    fourth = result.training_examples[3]
    assert fourth.data == {
        "intent": "restaurant_search",
        "entities": [
            {"start": 8, "end": 14, "value": "43er*+?df", "entity": "22_ab-34*3.A"}
        ],
    }
    assert fourth.text == "show me chines restaurants"


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


def test_read_complex_entity_format():
    r = MarkdownReader()

    md = """## intent:test
- I want to fly from [Berlin]{"entity": "city", "role": "to"} to [LA]{"entity": "city", "role": "from", "value": "Los Angeles"}
"""

    training_data = r.reads(md)

    assert "city" in training_data.entities
    assert training_data.entity_synonyms.get("LA") == "Los Angeles"

    entities = training_data.training_examples[0].data["entities"]

    assert len(entities) == 2
    assert entities[0]["role"] == "to"
    assert entities[0]["value"] == "Berlin"
    assert entities[1]["role"] == "from"
    assert entities[1]["value"] == "Los Angeles"


def test_read_simple_entity_format():
    r = MarkdownReader()

    md = """## intent:test
- I want to fly from [Berlin](city) to [LA](city)
"""

    training_data = r.reads(md)

    assert "city" in training_data.entities
    assert training_data.entity_synonyms.get("LA") is None

    entities = training_data.training_examples[0].data["entities"]

    assert len(entities) == 2
    assert entities[0]["value"] == "Berlin"
    assert entities[1]["value"] == "LA"


def test_read_mixed_entity_format():
    r = MarkdownReader()

    md = """## intent:test
- I want to fly from [Berlin](city) to [LA]{"entity": "city", "role": "from", "value": "Los Angeles"}
"""
    training_data = r.reads(md)

    assert "city" in training_data.entities
    assert training_data.entity_synonyms.get("LA") == "Los Angeles"

    entities = training_data.training_examples[0].data["entities"]

    assert len(entities) == 2
    assert "role" not in entities[0]
    assert entities[0]["value"] == "Berlin"
    assert entities[1]["role"] == "from"
    assert entities[1]["value"] == "Los Angeles"


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
def test_dump_trainable_entities(entity: Dict[Text, Any], expected_output: Text):
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
