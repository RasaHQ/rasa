from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

import numpy as np
import pytest
import spacy

from rasa_nlu.training_data import TrainingData


def test_crf_extractor(spacy_nlp_en):
    from rasa_nlu.extractors.crf_entity_extractor import CRFEntityExtractor
    ext = CRFEntityExtractor()
    examples = [{
                "text": "anywhere in the west",
                "intent": "restaurant_search",
                "entities": [
                              {
                                "start": 16,
                                "end": 20,
                                "value": "west",
                                "entity": "location"
                              }
                            ]
                },
                {
                "text": "central indian restaurant",
                "intent": "restaurant_search",
                "entities": [
                              {
                                "start": 0,
                                "end": 7,
                                "value": "central",
                                "entity": "location"
                              }
                            ]
                }]
    ext.train(TrainingData(entity_examples_only=examples), spacy_nlp_en, True, ext.crf_features)
    crf_format = ext._from_text_to_crf(u'anywhere in the west', spacy_nlp_en)
    assert([word[0] for word in crf_format] == [u'anywhere', u'in', u'the', u'west'])
    feats = ext._sentence_to_features(crf_format)
    assert(u'BOS' in feats[0])
    assert(u'EOS' in feats[-1])
    assert(u'0:low:in' in feats[1])
    ext.extract_entities(u'anywhere in the west', spacy_nlp_en)
