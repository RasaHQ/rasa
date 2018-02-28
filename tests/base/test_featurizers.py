from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import numpy as np
import pytest

from rasa_nlu import training_data
from rasa_nlu.tokenizers.mitie_tokenizer import MitieTokenizer
from rasa_nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa_nlu.training_data import Message


@pytest.mark.parametrize("sentence, expected", [
    ("hey how are you today", [-0.19649599, 0.32493639, -0.37408298, -0.10622784, 0.062756])
])
def test_spacy_featurizer(sentence, expected, spacy_nlp):
    from rasa_nlu.featurizers import spacy_featurizer
    doc = spacy_nlp(sentence)
    vecs = spacy_featurizer.features_for_doc(doc)
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
    repetition_factor = 5  # ensures that during random sampling of the ngram CV we don't end up with a one-class-split
    labeled_sentences = [
                            Message("heyheyheyhey", {"intent": "greet", "text_features": [0.5]}),
                            Message("howdyheyhowdy", {"intent": "greet", "text_features": [0.5]}),
                            Message("heyhey howdyheyhowdy", {"intent": "greet", "text_features": [0.5]}),
                            Message("howdyheyhowdy heyhey", {"intent": "greet", "text_features": [0.5]}),
                            Message("astalavistasista", {"intent": "goodby", "text_features": [0.5]}),
                            Message("astalavistasista sistala", {"intent": "goodby", "text_features": [0.5]}),
                            Message("sistala astalavistasista", {"intent": "goodby", "text_features": [0.5]}),
                        ] * repetition_factor

    for m in labeled_sentences:
        m.set("spacy_doc", spacy_nlp(m.text))

    ftr.min_intent_examples_for_ngram_classification = 2
    ftr.train_on_sentences(labeled_sentences,
                           max_number_of_ngrams=10)
    assert len(ftr.all_ngrams) > 0
    assert ftr.best_num_ngrams > 0


@pytest.mark.parametrize("sentence, expected, labeled_tokens", [
    ("hey how are you today", [0., 1.], [0]),
    ("hey 123 how are you", [1., 1.], [0, 1]),
    ("blah balh random eh", [0., 0.], []),
    ("looks really like 123 today", [1., 0.], [3]),
])
def test_regex_featurizer(sentence, expected, labeled_tokens, spacy_nlp):
    from rasa_nlu.featurizers.regex_featurizer import RegexFeaturizer
    patterns = [
        {"pattern": '[0-9]+', "name": "number", "usage": "intent"},
        {"pattern": '\\bhey*', "name": "hello", "usage": "intent"}
    ]
    ftr = RegexFeaturizer(patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer()
    message = Message(sentence)
    message.set("spacy_doc", spacy_nlp(sentence))
    tokenizer.process(message)

    result = ftr.features_for_patterns(message)
    assert np.allclose(result, expected, atol=1e-10)
    assert len(message.get("tokens", [])) > 0  # the tokenizer should have added tokens
    for i, token in enumerate(message.get("tokens")):
        if i in labeled_tokens:
            assert token.get("pattern") in [0, 1]
        else:
            assert token.get("pattern") is None  # if the token is not part of a regex the pattern should not be set


def test_spacy_featurizer_casing(spacy_nlp):
    from rasa_nlu.featurizers import spacy_featurizer

    # if this starts failing for the default model, we should think about
    # removing the lower casing the spacy nlp component does when it
    # retrieves vectors. For compressed spacy models (e.g. models
    # ending in _sm) this test will most likely fail.

    td = training_data.load_data('data/examples/rasa/demo-rasa.json')
    for e in td.intent_examples:
        doc = spacy_nlp(e.text)
        doc_capitalized = spacy_nlp(e.text.capitalize())

        vecs = spacy_featurizer.features_for_doc(doc)
        vecs_capitalized = spacy_featurizer.features_for_doc(doc_capitalized)

        assert np.allclose(vecs, vecs_capitalized, atol=1e-5), \
            "Vectors are unequal for texts '{}' and '{}'".format(
                    e.text, e.text.capitalize())
