# -*- coding: UTF-8 -*-
from rasa_nlu.training_data import TrainingData


def test_luis_mitie():
    td = TrainingData('data/examples/luis/demo-restaurants.json', 'mitie', 'en')
    assert td.fformat == 'luis'
    # some more assertions


def test_wit_spacy():
    td = TrainingData('data/examples/wit/demo-flights.json', 'spacy_sklearn', 'en')
    assert td.fformat == 'wit'


def test_rasa_whitespace():
    td = TrainingData('data/examples/rasa/demo-rasa.json', '', 'en')
    assert td.fformat == 'rasa_nlu'


def test_api_mitie():
    td = TrainingData('data/examples/api/', 'mitie', 'en')
    assert td.fformat == 'api'

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
    filename = 'tmp_training_data.json'
    with open(filename, 'w') as f:
        f.write(data.encode("utf-8"))
    td = TrainingData(filename, 'mitie', 'en')
    assert len(td.entity_examples) == 1
    example = td.entity_examples[0]
    entities = example["entities"]
    assert len(entities) == 1
    entity = entities[0]
    assert entity["value"] == u"ßäæ ?€ö)"
    assert entity["start"] == 19
    assert entity["end"] == 27
    assert entity["entity"] == "description"
