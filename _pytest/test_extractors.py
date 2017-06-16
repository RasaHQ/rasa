from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_nlu.training_data import TrainingData


def test_crf_extractor(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    examples = [
        {
            "text": "anywhere in the west",
            "intent": "restaurant_search",
            "entities": [{"start": 16, "end": 20, "value": "west", "entity": "location"}]
        },
        {
            "text": "central indian restaurant",
            "intent": "restaurant_search",
            "entities": [{"start": 0, "end": 7, "value": "central", "entity": "location"}]
        }]
    ext.train(TrainingData(training_examples=examples), spacy_nlp, True, ext.crf_features)
    crf_format = ext._from_text_to_crf('anywhere in the west', spacy_nlp)
    assert ([word[0] for word in crf_format] == ['anywhere', 'in', 'the', 'west'])
    feats = ext._sentence_to_features(crf_format)
    assert ('BOS' in feats[0])
    assert ('EOS' in feats[-1])
    assert ('0:low:in' in feats[1])
    ext.extract_entities('anywhere in the west', spacy_nlp)


def test_crf_json_from_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = True
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(spacy_nlp(sentence), ['O', 'O', 'O', 'B-what', 'L-what', 'B-where', 'I-where', 'L-where'])
    assert len(r) == 2, "There should be two entities"
    assert r[0] == {u'start': 9, u'end': 22, u'value': u'home cleaning', u'entity': u'what'}
    assert r[1] == {u'start': 23, u'end': 31, u'value': u'close-by', u'entity': u'where'}


def test_crf_json_from_non_BILOU(spacy_nlp):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    ext.BILOU_flag = False
    sentence = u"I need a home cleaning close-by"
    r = ext._from_crf_to_json(spacy_nlp(sentence), ['O', 'O', 'O', 'what', 'what', 'where', 'where', 'where'])
    assert len(r) == 5, "There should be five entities"  # non BILOU will split multi-word entities - hence 5
    assert r[0] == {u'start': 9, u'end': 13, u'value': u'home', u'entity': u'what'}
    assert r[1] == {u'start': 14, u'end': 22, u'value': u'cleaning', u'entity': u'what'}
    assert r[2] == {u'start': 23, u'end': 28, u'value': u'close', u'entity': u'where'}
    assert r[3] == {u'start': 28, u'end': 29, u'value': u'-', u'entity': u'where'}
    assert r[4] == {u'start': 29, u'end': 31, u'value': u'by', u'entity': u'where'}
