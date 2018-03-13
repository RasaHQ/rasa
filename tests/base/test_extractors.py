# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.extractors.spacy_entity_extractor import SpacyEntityExtractor
from rasa_nlu.training_data import TrainingData, Message
from tests import utilities


def test_crf_extractor(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    examples = [
        Message("anywhere in the west", {
            "intent": "restaurant_search",
            "entities": [{"start": 16, "end": 20, "value": "west", "entity": "location"}],
            "spacy_doc": spacy_nlp("anywhere in the west")
        }),
        Message("central indian restaurant", {
            "intent": "restaurant_search",
            "entities": [
                {"start": 0, "end": 7, "value": "central", "entity": "location", "extractor": "random_extractor"},
                {"start": 8, "end": 14, "value": "indian", "entity": "cuisine", "extractor": "ner_crf"}
            ],
            "spacy_doc": spacy_nlp("central indian restaurant")
        })]
    config = {"ner_crf": {"BILOU_flag": True, "features": ext.crf_features}}
    ext.train(TrainingData(training_examples=examples), config)
    sentence = 'anywhere in the west'
    crf_format = ext._from_text_to_crf(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}))
    assert [word[0] for word in crf_format] == ['anywhere', 'in', 'the', 'west']
    feats = ext._sentence_to_features(crf_format)
    assert 'BOS' in feats[0]
    assert 'EOS' in feats[-1]
    assert feats[1]['0:low'] == "in"
    sentence = 'anywhere in the west'
    ext.extract_entities(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}))
    filtered = ext.filter_trainable_entities(examples)
    assert filtered[0].get('entities') == [
        {"start": 16, "end": 20, "value": "west", "entity": "location"}
    ], 'Entity without extractor remains'
    assert filtered[1].get('entities') == [
        {"start": 8, "end": 14, "value": "indian", "entity": "cuisine", "extractor": "ner_crf"}
    ], 'Only ner_crf entity annotation remains'
    assert examples[1].get('entities')[0] == {
        "start": 0, "end": 7, "value": "central", "entity": "location", "extractor": "random_extractor"
    }, 'Original examples are not mutated'


def test_crf_json_from_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = True
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}),
                              ['O', 'O', 'O', 'B-what', 'L-what', 'B-where', 'I-where', 'L-where'])
    assert len(r) == 2, "There should be two entities"
    assert r[0] == {u'start': 9, u'end': 22, u'value': u'home cleaning', u'entity': u'what'}
    assert r[1] == {u'start': 23, u'end': 31, u'value': u'close-by', u'entity': u'where'}


def test_crf_json_from_non_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = False
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(Message(sentence, {"spacy_doc": spacy_nlp(sentence)}),
                              ['O', 'O', 'O', 'what', 'what', 'where', 'where', 'where'])
    assert len(r) == 5, "There should be five entities"  # non BILOU will split multi-word entities - hence 5
    assert r[0] == {u'start': 9, u'end': 13, u'value': u'home', u'entity': u'what'}
    assert r[1] == {u'start': 14, u'end': 22, u'value': u'cleaning', u'entity': u'what'}
    assert r[2] == {u'start': 23, u'end': 28, u'value': u'close', u'entity': u'where'}
    assert r[3] == {u'start': 28, u'end': 29, u'value': u'-', u'entity': u'where'}
    assert r[4] == {u'start': 29, u'end': 31, u'value': u'by', u'entity': u'where'}


def test_duckling_entity_extractor(component_builder):
    _config = utilities.base_test_conf("all_components")
    _config["duckling_dimensions"] = ["time"]
    duckling = component_builder.create_component("ner_duckling", _config)
    message = Message("Today is the 5th of May. Let us meet tomorrow.")
    duckling.process(message)
    entities = message.get("entities")
    assert len(entities) == 3

    # Test duckling with a defined date
    message = Message("Let us meet tomorrow.", time="1381536182000")  # 1381536182000 == 2013/10/12 02:03:02
    duckling.process(message)
    entities = message.get("entities")
    assert len(entities) == 1
    assert entities[0]["text"] == "tomorrow"
    assert entities[0]["value"] == "2013-10-13T00:00:00.000Z"


def test_duckling_entity_extractor_and_synonyms(component_builder):
    _config = utilities.base_test_conf("all_components")
    _config["duckling_dimensions"] = ["number"]
    duckling = component_builder.create_component("ner_duckling", _config)
    synonyms = component_builder.create_component("ner_synonyms", _config)
    message = Message("He was 6 feet away")
    duckling.process(message)
    synonyms.process(message)  # checks that the synonym processor can handle entities that have int values
    assert message is not None


def test_unintentional_synonyms_capitalized(component_builder):
    _config = utilities.base_test_conf("all_components")
    ner_syn = component_builder.create_component("ner_synonyms", _config)
    examples = [
        Message("Any Mexican restaurant will do", {
            "intent": "restaurant_search",
            "entities": [{"start": 4, "end": 11, "value": "Mexican", "entity": "cuisine"}]
        }),
        Message("I want Tacos!", {
            "intent": "restaurant_search",
            "entities": [{"start": 7, "end": 12, "value": "Mexican", "entity": "cuisine"}]
        })
    ]
    ner_syn.train(TrainingData(training_examples=examples), _config)
    assert ner_syn.synonyms.get("mexican") is None
    assert ner_syn.synonyms.get("tacos") == "Mexican"


def test_spacy_ner_extractor(spacy_nlp):
    ext = SpacyEntityExtractor()
    example = Message("anywhere in the West", {
            "intent": "restaurant_search",
            "entities": [],
            "spacy_doc": spacy_nlp("anywhere in the west")})

    ext.process(example, spacy_nlp=spacy_nlp)

    assert len(example.get("entities", [])) == 1
    assert example.get("entities")[0] == {
        u'start': 16, u'extractor': u'ner_spacy',
        u'end': 20, u'value': u'West', u'entity': u'LOC'}
