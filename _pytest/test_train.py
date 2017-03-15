# -*- coding: utf-8 -*-
import tempfile

import pytest

from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME, MITIE_SKLEARN_BACKEND_NAME
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME
import utilities


def base_test_conf(backend):
    return {
        'write': utilities.temp_log_file_location(),
        'port': 5022,
        "backend": backend,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    MITIE_SKLEARN_BACKEND_NAME,
    SPACY_BACKEND_NAME,
    ])
def test_train_backend(backend_name, spacy_nlp_en):
    _config = base_test_conf(backend_name)
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    loaded = utilities.load_interpreter_for_model(spacy_nlp_en, _config, persisted_path)
    assert loaded.extractor is not None
    assert loaded.classifier is not None


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    MITIE_SKLEARN_BACKEND_NAME,
    SPACY_BACKEND_NAME,
])
def test_train_backend_noents(backend_name, spacy_nlp_en):
    _config = base_test_conf(backend_name)
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.entity_extractor is None
    assert trained.intent_classifier is not None
    loaded = utilities.load_interpreter_for_model(spacy_nlp_en, _config, persisted_path)
    assert loaded.extractor is None
    assert loaded.classifier is not None


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    SPACY_BACKEND_NAME,
])
def test_train_backend_multithread(backend_name, spacy_nlp_en):
    # basic conf
    _config = base_test_conf(backend_name)
    _config['num_threads'] = 2
    (trained, persisted_path) = utilities.run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    loaded = utilities.load_interpreter_for_model(spacy_nlp_en, _config, persisted_path)
    assert loaded.extractor is not None
    assert loaded.classifier is not None


def test_train_spacy_sklearn_finetune_ner():
    # basic conf
    _config = base_test_conf(SPACY_BACKEND_NAME)
    _config['fine_tune_spacy_ner'] = True
    (trained, _) = utilities.run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    doc = trained.nlp(u"I am living in New York now.")
    entities = trained.entity_extractor.extract_entities(doc)
    # Although the model is trained on restaurant entities, we can use the entities (`GPE`, `DATE`)
    # from spacy since we are fine tuning. This should even be the case if the rasa-entity training data changes!
    assert {u'start': 15, u'end': 23, u'value': u'New York', u'entity': u'GPE'} in entities
