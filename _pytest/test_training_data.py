# -*- coding: utf-8 -*-
import os
import tempfile
import spacy

from rasa_nlu.training_data import TrainingData
from rasa_nlu.trainers import mitie_trainer_utils


def test_luis_mitie():
    td = TrainingData('data/examples/luis/demo-restaurants.json', 'mitie')
    assert td.fformat == 'luis'
    # some more assertions


def test_wit_spacy(spacy_nlp_en):
    td = TrainingData('data/examples/wit/demo-flights.json', 'spacy_sklearn', nlp=spacy_nlp_en)
    assert td.fformat == 'wit'


def test_rasa_whitespace():
    td = TrainingData('data/examples/rasa/demo-rasa.json', '', 'en')
    assert td.fformat == 'rasa_nlu'


def test_api_mitie():
    td = TrainingData('data/examples/api/', 'mitie', 'en')
    assert td.fformat == 'api'


def test_api_mitie_sklearn():
    td = TrainingData('data/examples/api/', 'mitie_sklearn', 'en')
    assert td.fformat == 'api'


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
        td = TrainingData(f.name, 'mitie', 'en')
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = mitie_trainer_utils.find_entity(entities[0], example["text"])
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
        td = TrainingData(f.name, 'mitie', 'en')
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        start, end = mitie_trainer_utils.find_entity(entities[0], example["text"])
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
        td = TrainingData(f.name, 'mitie', 'en')
        assert len(td.entity_examples) == 1
        example = td.entity_examples[0]
        entities = example["entities"]
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == u"ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"


def test_entities_synonyms():
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
            "value": "NYC"
          }
        ]
      },
      {
        "text": "show me flights to NYC",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 22,
            "value": "NYC"
          }
        ]
      }
    ]
  }
}"""
    with tempfile.NamedTemporaryFile(suffix="_tmp_training_data.json") as f:
        f.write(data.encode("utf-8"))
        f.flush()
        td = TrainingData(f.name, 'mitie', 'en')
        assert len(td.entity_synonyms) == 1
        assert td.entity_synonyms["new york city"] == "nyc"
