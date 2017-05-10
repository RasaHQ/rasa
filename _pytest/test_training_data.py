# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import tempfile

import io
import pytest
from jsonschema import ValidationError

from rasa_nlu.converters import load_data, validate_rasa_nlu_data
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor


def test_example_training_data_is_valid():
    with io.open('data/examples/rasa/demo-rasa.json', encoding="utf-8-sig") as f:
        data = json.loads(f.read())
    validate_rasa_nlu_data(data)


@pytest.mark.parametrize("invalid_data", [
    {"wrong_top_level": []},
    ["this is not a toplevel dict"],
    {"rasa_nlu_data": {"common_examples": [{"intent": "some example without text"}]}},
    {"rasa_nlu_data": {
        "common_examples": [{
            "text": "mytext", "entities": [{"start": "INVALID", "end": 0, "entity": "x"}]
        }]
    }},
])
def test_validation_is_throwing_exceptions(invalid_data):
    with pytest.raises(ValidationError):
        validate_rasa_nlu_data(invalid_data)


def test_luis_data():
    td = load_data('data/examples/luis/demo-restaurants.json')
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms == {}


def test_wit_data():
    td = load_data('data/examples/wit/demo-flights.json')
    assert td.entity_examples != []
    assert td.intent_examples == []
    assert td.entity_synonyms == {}


def test_rasa_data():
    td = load_data('data/examples/rasa/demo-rasa.json')
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert len(td.sorted_entity_examples()) >= len([e for e in td.entity_examples if e["entities"]])
    assert len(td.sorted_intent_examples()) == len(td.intent_examples)
    assert td.entity_synonyms == {}


def test_api_data():
    td = load_data('data/examples/api/')
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms != {}


def test_repeated_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "book a table today from 3 to 6 for 3 people",
        "intent": "unk",
        "entities": [
          {
            "entity": "description",
            "start": 35,
            "end": 36,
            "value": "3"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = MitieEntityExtractor.find_entity(entities[0], example["text"])
        assert start == 9
        assert end == 10


def test_multiword_entities():
    data = """
{
  "rasa_nlu_data": {
    "common_examples" : [
      {
        "text": "show me flights to New York City",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 32,
            "value": "New York City"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = MitieEntityExtractor.find_entity(entities[0], example["text"])
        assert start == 4
        assert end == 7


def test_nonascii_entities():
    data = """
{
  "luis_schema_version": "2.0",
  "utterances" : [
    {
      "text": "I am looking for a ßäæ ?€ö) item",
      "intent": "unk",
      "entities": [
        {
          "entity": "description",
          "startPos": 19,
          "endPos": 26
        }
      ]
    }
  ]
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = load_data(f.name)
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == "ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"
