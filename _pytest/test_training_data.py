# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tempfile

import pytest

from rasa_nlu.converters import load_data
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor


def test_luis_data_spacy():
    td = load_data('data/examples/luis/demo-restaurants.json', "en", "tokenizer_spacy")
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms == {}


def test_luis_data_mitie():
    td = load_data('data/examples/luis/demo-restaurants.json', "en", "tokenizer_mitie")
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms == {}


def test_luis_data_without_tokenizer():
    with pytest.raises(ValueError):
        load_data('data/examples/luis/demo-restaurants.json', "en")


def test_wit_data():
    td = load_data('data/examples/wit/demo-flights.json', "en")
    assert td.entity_examples != []
    assert td.intent_examples == []
    assert td.entity_synonyms == {}


def test_rasa_data():
    td = load_data('data/examples/rasa/demo-rasa.json', "en")
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert len(td.sorted_entity_examples()) >= len([e for e in td.entity_examples if e["entities"]])
    assert len(td.sorted_intent_examples()) == len(td.intent_examples)
    assert td.entity_synonyms == {}


def test_api_data():
    td = load_data('data/examples/api/', "en")
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms != {}


def test_repeated_entities():
    data = u"""
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
            "value": 3
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = load_data(f.name, "en")
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = MitieEntityExtractor.find_entity(entities[0], example["text"])
        assert start == 9
        assert end == 10


def test_multiword_entities():
    data = u"""
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
        td = load_data(f.name, "en")
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = MitieEntityExtractor.find_entity(entities[0], example["text"])
        assert start == 4
        assert end == 7


def test_nonascii_entities():
    data = u"""
{
  "luis_schema_version": "1.0",
  "utterances" : [
    {
      "text": "I am looking for a ßäæ ?€ö) item",
      "intent": "unk",
      "entities": [
        {
          "entity": "description",
          "startPos": 5,
          "endPos": 8
        }
      ]
    }
  ]
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = load_data(f.name, "en", luis_data_tokenizer="tokenizer_mitie")
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == u"ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"


# def test_entities_synonyms():
#     data = u"""
# {
#   "rasa_nlu_data": {
#     "common_examples" : [
#       {
#         "text": "show me flights to New York City",
#         "intent": "unk",
#         "entities": [
#           {
#             "entity": "destination",
#             "start": 19,
#             "end": 32,
#             "value": "NYC"
#           }
#         ]
#       },
#       {
#         "text": "show me flights to NYC",
#         "intent": "unk",
#         "entities": [
#           {
#             "entity": "destination",
#             "start": 19,
#             "end": 22,
#             "value": "NYC"
#           }
#         ]
#       }
#     ]
#   }
# }"""
#     with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
#         f.write(data.encode("utf-8"))
#         f.flush()
#         td = load_data(f.name, "en")
#         assert td.entity_synonyms["new york city"] == "nyc"
