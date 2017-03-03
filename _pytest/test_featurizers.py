import os

import numpy as np
import pytest

from rasa_nlu.tokenizers.mitie_tokenizer import MITIETokenizer


@pytest.mark.parametrize("sentence, language, expected", [
    (u"hey how are you today", "en", [-0.19649599, 0.32493639, -0.37408298, -0.10622784, 0.062756]),
    (u"hey wie geht es dir", "de", [-0.0518572, -0.13645099, 0.34630662, 0.29546982, -0.0153512]),
])
def test_spacy_featurizer(sentence, language, expected):
    import spacy
    from rasa_nlu.featurizers.spacy_featurizer import SpacyFeaturizer
    nlp = spacy.load(language, tagger=False, parser=False)
    ftr = SpacyFeaturizer()
    doc = nlp(sentence)
    vecs = ftr.features_for_doc(doc)
    assert np.allclose(doc.vector[:5], expected, atol=1e-5)
    assert np.allclose(vecs, doc.vector, atol=1e-5)


def test_mitie_featurizer():
    from rasa_nlu.featurizers.mitie_featurizer import MITIEFeaturizer

    filename = os.environ.get('MITIE_FILE')
    if not filename or not os.path.isfile(filename):
        filename = "data/total_word_feature_extractor.dat"

    ftr = MITIEFeaturizer(filename)
    sentence = "Hey how are you today"
    tokens = MITIETokenizer().tokenize(sentence)
    vecs = ftr.features_for_tokens(tokens)
    assert np.allclose(vecs[:5], np.array([0., -4.4551446, 0.26073121, -1.46632245, -1.84205751]), atol=1e-5)
