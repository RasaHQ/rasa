from typing import Any, Dict, Text

import numpy as np
import pytest

from rasa.nlu.utils.spacy_utils import SpacyModel, SpacyNLP
from rasa.shared.nlu.training_data import loading
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.featurizers.dense_featurizer.spacy_featurizer import SpacyFeaturizer
from rasa.nlu.constants import SPACY_DOCS
from rasa.shared.nlu.constants import TEXT, INTENT, RESPONSE


def create_spacy_featurizer(config: Dict[Text, Any]) -> SpacyFeaturizer:
    return SpacyFeaturizer(
        {**SpacyFeaturizer.get_default_config(), **config}, "spacy_featurizer"
    )


def test_spacy_featurizer_cls_vector(spacy_nlp):
    featurizer = create_spacy_featurizer({})

    sentence = "Hey how are you today"
    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))

    featurizer._set_spacy_features(message)

    seq_vecs, sen_vecs = message.get_dense_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    assert 5 == len(seq_vecs)
    assert 1 == len(sen_vecs)


@pytest.mark.parametrize("sentence", ["hey how are you today"])
def test_spacy_featurizer(sentence, spacy_nlp):

    ftr = create_spacy_featurizer({})

    doc = spacy_nlp(sentence)
    vecs = ftr._features_for_doc(doc)
    expected = [t.vector for t in doc]

    assert np.allclose(vecs, expected, atol=1e-5)


def test_spacy_training_sample_alignment(
    spacy_nlp_component: SpacyNLP, spacy_model: SpacyModel
):
    from spacy.tokens import Doc

    m1 = Message.build(text="I have a feeling", intent="feeling")
    m2 = Message.build(text="", intent="feeling")
    m3 = Message.build(text="I am the last message", intent="feeling")
    td = TrainingData(training_examples=[m1, m2, m3])

    attribute_docs = spacy_nlp_component._docs_for_training_data(spacy_model.model, td)

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


def test_spacy_intent_featurizer(
    spacy_nlp_component: SpacyNLP, spacy_model: SpacyModel
):
    td = loading.load_data("data/examples/rasa/demo-rasa.json")
    spacy_nlp_component.process_training_data(td, spacy_model)
    spacy_featurizer = create_spacy_featurizer({})
    spacy_featurizer.process_training_data(td)

    intent_features_exist = np.array(
        [
            True if example.get("intent_features") is not None else False
            for example in td.intent_examples
        ]
    )

    # no intent features should have been set
    assert not any(intent_features_exist)


def test_spacy_featurizer_sequence(spacy_nlp):
    sentence = "hey how are you today"
    doc = spacy_nlp(sentence)
    token_vectors = [t.vector for t in doc]

    ftr = create_spacy_featurizer({})

    greet = {TEXT: sentence, "intent": "greet", "text_features": [0.5]}

    message = Message(data=greet)
    message.set(SPACY_DOCS[TEXT], doc)

    ftr._set_spacy_features(message)

    seq_vecs, sen_vecs = message.get_dense_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    vecs = seq_vecs[0][:5]

    assert np.allclose(token_vectors[0][:5], vecs, atol=1e-4)
    assert sen_vecs is not None


def test_spacy_featurizer_default_case_insensitive(spacy_nlp_component):
    ftr = create_spacy_featurizer({})
    spacy_nlp = spacy_nlp_component.provide().model
    td = loading.load_data("data/examples/rasa/demo-rasa.json")
    for e in td.intent_examples:
        doc = spacy_nlp_component._doc_for_text(spacy_nlp, e.get(TEXT))
        doc_capitalized = spacy_nlp_component._doc_for_text(
            spacy_nlp, e.get(TEXT).capitalize()
        )

        vecs = ftr._features_for_doc(doc)
        vecs_capitalized = ftr._features_for_doc(doc_capitalized)

        assert np.allclose(
            vecs, vecs_capitalized, atol=1e-5
        ), "Vectors are unequal for texts '{}' and '{}'".format(
            e.get(TEXT), e.get(TEXT).capitalize()
        )


def test_spacy_featurizer_can_be_case_sensitive(spacy_case_sensitive_nlp_component):
    ftr = create_spacy_featurizer({})
    spacy_nlp = spacy_case_sensitive_nlp_component.provide().model
    td = loading.load_data("data/examples/rasa/demo-rasa.json")
    example_is_case_insentive = []
    for e in td.intent_examples:
        doc = spacy_case_sensitive_nlp_component._doc_for_text(spacy_nlp, e.get(TEXT))
        doc_capitalized = spacy_case_sensitive_nlp_component._doc_for_text(
            spacy_nlp, e.get(TEXT).capitalize()
        )

        vecs = ftr._features_for_doc(doc)
        vecs_capitalized = ftr._features_for_doc(doc_capitalized)

        example_is_case_insentive.append(np.allclose(vecs, vecs_capitalized, atol=1e-5))
    assert not all(example_is_case_insentive)


def test_spacy_featurizer_train(spacy_nlp):

    featurizer = create_spacy_featurizer({})

    sentence = "Hey how are you today"
    message = Message(data={TEXT: sentence})
    message.set(RESPONSE, sentence)
    message.set(INTENT, "intent")
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    message.set(SPACY_DOCS[RESPONSE], spacy_nlp(sentence))

    featurizer.process_training_data(TrainingData([message]))

    seq_vecs, sen_vecs = message.get_dense_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    assert 5 == len(seq_vecs)
    assert 1 == len(sen_vecs)

    seq_vecs, sen_vecs = message.get_dense_features(RESPONSE, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    assert 5 == len(seq_vecs)
    assert 1 == len(sen_vecs)

    seq_vecs, sen_vecs = message.get_dense_features(INTENT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    assert seq_vecs is None
    assert sen_vecs is None


def test_spacy_featurizer_using_empty_model():
    import spacy

    sentence = "This test is using an empty spaCy model"

    model = spacy.blank("en")
    doc = model(sentence)

    ftr = create_spacy_featurizer({})

    message = Message(data={TEXT: sentence})
    message.set(SPACY_DOCS[TEXT], doc)

    ftr._set_spacy_features(message)

    seq_vecs, sen_vecs = message.get_dense_features(TEXT, [])
    if seq_vecs:
        seq_vecs = seq_vecs.features
    if sen_vecs:
        sen_vecs = sen_vecs.features

    assert seq_vecs is None
    assert sen_vecs is None
