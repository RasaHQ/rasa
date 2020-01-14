import numpy as np
import pytest

from rasa.nlu import training_data
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.constants import (
    SPACY_DOCS,
    TEXT_ATTRIBUTE,
    DENSE_FEATURE_NAMES,
    RESPONSE_ATTRIBUTE,
    INTENT_ATTRIBUTE,
)


def test_spacy_featurizer_cls_vector(spacy_nlp):
    featurizer = SpacyFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today"
    message = Message(sentence)
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(sentence))

    featurizer._set_spacy_features(message)

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE])

    expected = np.array([-0.28451, 0.31007, -0.57039, -0.073056, -0.17322])
    expected_cls = np.array([-0.196496, 0.3249364, -0.37408298, -0.10622784, 0.062756])

    assert 6 == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)


@pytest.mark.parametrize("sentence", ["hey how are you today"])
def test_spacy_featurizer(sentence, spacy_nlp):

    ftr = SpacyFeaturizer.create({}, RasaNLUModelConfig())

    doc = spacy_nlp(sentence)
    vecs = ftr._features_for_doc(doc)
    expected = [t.vector for t in doc]

    assert np.allclose(vecs, expected, atol=1e-5)


def test_spacy_training_sample_alignment(spacy_nlp_component):
    from spacy.tokens import Doc

    m1 = Message.build(text="I have a feeling", intent="feeling")
    m2 = Message.build(text="", intent="feeling")
    m3 = Message.build(text="I am the last message", intent="feeling")
    td = TrainingData(training_examples=[m1, m2, m3])

    attribute_docs = spacy_nlp_component.docs_for_training_data(td)

    assert isinstance(attribute_docs["text"][0], Doc)
    assert isinstance(attribute_docs["text"][1], Doc)
    assert isinstance(attribute_docs["text"][2], Doc)

    assert [t.text for t in attribute_docs["text"][0]] == ["i", "have", "a", "feeling"]
    assert [t.text for t in attribute_docs["text"][1]] == []
    assert [t.text for t in attribute_docs["text"][2]] == [
        "i",
        "am",
        "the",
        "last",
        "message",
    ]


def test_spacy_intent_featurizer(spacy_nlp_component):
    from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.train(td, config=None)
    spacy_featurizer = SpacyFeaturizer()
    spacy_featurizer.train(td, config=None)

    intent_features_exist = np.array(
        [
            True if example.get("intent_features") is not None else False
            for example in td.intent_examples
        ]
    )

    # no intent features should have been set
    assert not any(intent_features_exist)


@pytest.mark.parametrize(
    "sentence, expected",
    [("hey how are you today", [-0.28451, 0.31007, -0.57039, -0.073056, -0.17322])],
)
def test_spacy_featurizer_sequence(sentence, expected, spacy_nlp):
    from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer

    doc = spacy_nlp(sentence)
    token_vectors = [t.vector for t in doc]

    ftr = SpacyFeaturizer.create({}, RasaNLUModelConfig())

    greet = {"intent": "greet", "text_features": [0.5]}

    message = Message(sentence, greet)
    message.set("spacy_doc", doc)

    ftr._set_spacy_features(message)

    vecs = message.get("text_dense_features")[0][:5]

    assert np.allclose(token_vectors[0][:5], vecs, atol=1e-4)
    assert np.allclose(vecs, expected, atol=1e-4)


def test_spacy_featurizer_casing(spacy_nlp):
    from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer

    # if this starts failing for the default model, we should think about
    # removing the lower casing the spacy nlp component does when it
    # retrieves vectors. For compressed spacy models (e.g. models
    # ending in _sm) this test will most likely fail.

    ftr = SpacyFeaturizer.create({}, RasaNLUModelConfig())

    td = training_data.load_data("data/examples/rasa/demo-rasa.json")
    for e in td.intent_examples:
        doc = spacy_nlp(e.text)
        doc_capitalized = spacy_nlp(e.text.capitalize())

        vecs = ftr._features_for_doc(doc)
        vecs_capitalized = ftr._features_for_doc(doc_capitalized)

        assert np.allclose(
            vecs, vecs_capitalized, atol=1e-5
        ), "Vectors are unequal for texts '{}' and '{}'".format(
            e.text, e.text.capitalize()
        )


def test_spacy_featurizer_train(spacy_nlp):

    featurizer = SpacyFeaturizer.create({}, RasaNLUModelConfig())

    sentence = "Hey how are you today"
    message = Message(sentence)
    message.set(RESPONSE_ATTRIBUTE, sentence)
    message.set(INTENT_ATTRIBUTE, "intent")
    message.set(SPACY_DOCS[TEXT_ATTRIBUTE], spacy_nlp(sentence))
    message.set(SPACY_DOCS[RESPONSE_ATTRIBUTE], spacy_nlp(sentence))

    featurizer.train(TrainingData([message]), RasaNLUModelConfig())

    expected = np.array([-0.28451, 0.31007, -0.57039, -0.073056, -0.17322])
    expected_cls = np.array([-0.196496, 0.3249364, -0.37408298, -0.10622784, 0.062756])

    vecs = message.get(DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE])

    assert 6 == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)

    vecs = message.get(DENSE_FEATURE_NAMES[RESPONSE_ATTRIBUTE])

    assert 6 == len(vecs)
    assert np.allclose(vecs[0][:5], expected, atol=1e-5)
    assert np.allclose(vecs[-1][:5], expected_cls, atol=1e-5)

    vecs = message.get(DENSE_FEATURE_NAMES[INTENT_ATTRIBUTE])

    assert vecs is None
