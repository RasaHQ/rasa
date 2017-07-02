from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os

import numpy as np
import pytest

from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer


@pytest.mark.parametrize("sentence, expected", [
    ("hey how are you today", [-0.19649599, 0.32493639, -0.37408298, -0.10622784, 0.062756])
])
def test_spacy_featurizer(sentence, expected, spacy_nlp):
    from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
    ftr = SpacyFeaturizer()
    doc = spacy_nlp(sentence)
    vecs = ftr.features_for_doc(doc, spacy_nlp)
    assert np.allclose(doc.vector[:5], expected, atol=1e-5)
    assert np.allclose(vecs, doc.vector, atol=1e-5)


def test_mitie_featurizer(mitie_feature_extractor, default_config):
    from rasa_nlu.featurizers.mitie_featurizer import MitieFeaturizer

    default_config["mitie_file"] = os.environ.get('MITIE_FILE')
    if not default_config["mitie_file"] or not os.path.isfile(default_config["mitie_file"]):
        default_config["mitie_file"] = os.path.join("data", "total_word_feature_extractor.dat")

    ftr = MitieFeaturizer.load()
    sentence = "Hey how are you today"
    tokens = MitieTokenizer().tokenize(sentence)
    vecs = ftr.features_for_tokens(tokens, mitie_feature_extractor)
    assert np.allclose(vecs[:5], np.array([0., -4.4551446, 0.26073121, -1.46632245, -1.84205751]), atol=1e-5)


def test_ngram_featurizer(spacy_nlp):
    from rasa_nlu.featurizers.ngram_featurizer import NGramFeaturizer
    ftr = NGramFeaturizer()
    repetition_factor = 5   # ensures that during random sampling of the ngram CV we don't end up with a one-class-split
    labeled_sentences = {
        "heyheyheyhey": "greet",
        "howdyheyhowdy": "greet",
        "heyhey howdyheyhowdy": "greet",
        "howdyheyhowdy heyhey": "greet",
        "astalavistasista": "goodby",
        "astalavistasista sistala": "goodby",
        "sistala astalavistasista": "goodby",
    }
    ftr.min_intent_examples_for_ngram_classification = 2
    ftr.train_on_sentences(list(labeled_sentences.keys()) * repetition_factor,
                           list(labeled_sentences.values()) * repetition_factor,
                           spacy_nlp,
                           max_number_of_ngrams=10,
                           intent_features=[[0.5]] * len(labeled_sentences) * repetition_factor)
    assert len(ftr.all_ngrams) > 0
    assert ftr.best_num_ngrams > 0
