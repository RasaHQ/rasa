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
    ext.train(TrainingData(entity_examples_only=examples), spacy_nlp, True, ext.crf_features)
    crf_format = ext._from_text_to_crf('anywhere in the west', spacy_nlp)
    assert ([word[0] for word in crf_format] == ['anywhere', 'in', 'the', 'west'])
    feats = ext._sentence_to_features(crf_format)
    assert ('BOS' in feats[0])
    assert ('EOS' in feats[-1])
    assert ('0:low:in' in feats[1])
    ext.extract_entities('anywhere in the west', spacy_nlp)
