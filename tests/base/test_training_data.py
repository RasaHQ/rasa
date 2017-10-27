# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import tempfile

import pytest
from jsonschema import ValidationError

from rasa_nlu.convert import convert_training_data
from rasa_nlu.converters import load_data, validate_rasa_nlu_data
from rasa_nlu.extractors.mitie_entity_extractor import MitieEntityExtractor
from rasa_nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


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
    assert td.intent_examples != []
    assert td.entity_synonyms == {}


def test_rasa_data():
    td = load_data('data/examples/rasa/demo-rasa.json')
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert len(td.sorted_entity_examples()) >= len([e for e in td.entity_examples if e.get("entities")])
    assert len(td.sorted_intent_examples()) == len(td.intent_examples)
    assert td.entity_synonyms == {u'Chines': u'chinese', u'Chinese': u'chinese', u'chines': u'chinese',
                                  u'vegg': u'vegetarian', u'veggie': u'vegetarian'}


def test_dialogflow_data():
    td = load_data('data/examples/dialogflow/')
    assert td.entity_examples != []
    assert td.intent_examples != []
    assert td.entity_synonyms != {}


def test_markdown_data():
    td = load_data('data/examples/rasa/demo-rasa.md')
    assert len(td.sorted_entity_examples()) >= len([e for e in td.entity_examples if e.get("entities")])
    assert len(td.sorted_intent_examples()) == len(td.intent_examples)
    assert td.entity_synonyms == {u'Chines': u'chinese', u'Chinese': u'chinese', u'chines': u'chinese',
                                  u'vegg': u'vegetarian', u'veggie': u'vegetarian'}


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
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example.text)
        start, end = MitieEntityExtractor.find_entity(entities[0], example.text, tokens)
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
        entities = example.get("entities")
        assert len(entities) == 1
        tokens = WhitespaceTokenizer().tokenize(example.text)
        start, end = MitieEntityExtractor.find_entity(entities[0], example.text, tokens)
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
        entities = example.get("entities")
        assert len(entities) == 1
        entity = entities[0]
        assert entity["value"] == "ßäæ ?€ö)"
        assert entity["start"] == 19
        assert entity["end"] == 27
        assert entity["entity"] == "description"


def test_entities_synonyms():
    data = u"""
{
  "rasa_nlu_data": {
    "entity_synonyms": [
      {
        "value": "nyc",
        "synonyms": ["New York City", "nyc", "the big apple"]
      }
    ],
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
        "text": "show me flights to nyc",
        "intent": "unk",
        "entities": [
          {
            "entity": "destination",
            "start": 19,
            "end": 22,
            "value": "nyc"
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
        assert td.entity_synonyms["New York City"] == "nyc"


def cmp_message_list(firsts, seconds):
    assert len(firsts) == len(seconds), "Message lists have unequal length"


def cmp_dict_list(firsts, seconds):
    if len(firsts) != len(seconds):
        return False

    for a in firsts:
        for idx, b in enumerate(seconds):
            if hash(a) == hash(b):
                del seconds[idx]
                break
        else:
            assert False, "Failed to find message {} in {}".format(a.text, ", ".join([e.text for e in seconds]))
    return not seconds


@pytest.mark.parametrize("data_file,gold_standard_file,output_format,language", [
    ("data/examples/wit/demo-flights.json", "data/test/wit_converted_to_rasa.json", "json", None),
    ("data/examples/luis/demo-restaurants.json", "data/test/luis_converted_to_rasa.json", "json", None),
    ("data/examples/dialogflow/", "data/test/dialogflow_en_converted_to_rasa.json", "json", "en"),
    ("data/examples/dialogflow/", "data/test/dialogflow_es_converted_to_rasa.json", "json", "es"),
    ("data/examples/rasa/demo-rasa.md", "data/test/md_converted_to_json.json", "json", None),
    ("data/examples/rasa/demo-rasa.json", "data/test/json_converted_to_md.md", "md", None)
])
def test_training_data_conversion(tmpdir, data_file, gold_standard_file, output_format, language):
    out_path = tmpdir.join("rasa_nlu_data.json")
    convert_training_data(data_file, out_path.strpath, output_format, language)
    td = load_data(out_path.strpath, language)
    assert td.entity_examples != []
    assert td.intent_examples != []

    gold_standard = load_data(gold_standard_file, language)
    cmp_message_list(td.entity_examples, gold_standard.entity_examples)
    cmp_message_list(td.intent_examples, gold_standard.intent_examples)
    assert td.entity_synonyms == gold_standard.entity_synonyms

    # converting the converted file back to original file format and performing the same tests
    rto_path = tmpdir.join("data_in_original_format.txt")
    convert_training_data(out_path.strpath, rto_path.strpath, 'json', language)
    rto = load_data(rto_path.strpath, language)
    cmp_message_list(gold_standard.entity_examples, rto.entity_examples)
    cmp_message_list(gold_standard.intent_examples, rto.intent_examples)
    assert gold_standard.entity_synonyms == rto.entity_synonyms

    # If the above assert fails - this can be used to dump to the file and diff using git
    # with io.open(gold_standard_file) as f:
    #     f.write(td.as_json(indent=2))
