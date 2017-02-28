# -*- coding: utf-8 -*-
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.train import do_train
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME, MITIE_SKLEARN_BACKEND_NAME
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME
from rasa_nlu.server import __create_interpreter


def run_train(_config):
    config = RasaNLUConfig(cmdline_args=_config)
    (trained, path) = do_train(config)
    return trained, path


def load_interpreter_for_model(config, persisted_path):
    config['server_model_dir'] = persisted_path
    return __create_interpreter(config)


def base_test_conf(backend):
    return {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": backend,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }


def temp_log_file_location():
    return tempfile.mkstemp(suffix="_rasa_nlu_logs.json")[1]


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    MITIE_SKLEARN_BACKEND_NAME,
    SPACY_BACKEND_NAME,
    ])
def test_train_backend(backend_name):
    _config = base_test_conf(backend_name)
    (trained, persisted_path) = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    loaded = load_interpreter_for_model(_config, persisted_path)
    assert loaded.extractor is not None
    assert loaded.classifier is not None


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    MITIE_SKLEARN_BACKEND_NAME,
    SPACY_BACKEND_NAME,
])
def test_train_backend_noents(backend_name):
    _config = base_test_conf(backend_name)
    _config['data'] = "./data/examples/rasa/demo-rasa-noents.json"
    (trained, persisted_path) = run_train(_config)
    assert trained.entity_extractor is None
    assert trained.intent_classifier is not None
    loaded = load_interpreter_for_model(_config, persisted_path)
    assert loaded.extractor is None
    assert loaded.classifier is not None


@pytest.mark.parametrize("backend_name", [
    MITIE_BACKEND_NAME,
    SPACY_BACKEND_NAME,
])
def test_train_backend_multithread(backend_name):
    # basic conf
    _config = base_test_conf(backend_name)
    _config['num_threads'] = 2
    (trained, persisted_path) = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    loaded = load_interpreter_for_model(_config, persisted_path)
    assert loaded.extractor is not None
    assert loaded.classifier is not None


def test_train_spacy_sklearn_finetune_ner():
    # basic conf
    _config = base_test_conf(SPACY_BACKEND_NAME)
    _config['fine_tune_spacy_ner'] = True
    (trained, _) = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
    doc = trained.nlp(u"I am living in New York now.")
    entities = trained.entity_extractor.extract_entities(doc)
    # Although the model is trained on restaurant entities, we can use the entities (`GPE`, `DATE`)
    # from spacy since we are fine tuning. This should even be the case if the rasa-entity training data changes!
    assert {u'start': 15, u'end': 23, u'value': u'New York', u'entity': u'GPE'} in entities
    assert {u'start': 24, u'end': 27, u'value': u'now', u'entity': u'DATE'} in entities
