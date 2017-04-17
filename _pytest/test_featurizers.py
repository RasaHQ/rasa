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
