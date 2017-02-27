# -*- coding: utf-8 -*-
import tempfile

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.train import do_train
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME


def run_train(_config):
    config = RasaNLUConfig(cmdline_args=_config)
    return do_train(config)


def temp_log_file_location():
    return tempfile.mkstemp(suffix="_rasa_nlu_logs.json")[1]


def test_train_mitie():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": MITIE_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None


def test_train_mitie_sklearn():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": "mitie_sklearn",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None


def test_train_mitie_noents():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": MITIE_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is None
    assert trained.intent_classifier is not None


def test_train_mitie_multithread():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": MITIE_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "num_threads": 2,
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None


def test_train_spacy_sklearn():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": SPACY_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None


def test_train_spacy_sklearn_noents():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": SPACY_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is None
    assert trained.intent_classifier is not None


def test_train_spacy_sklearn_finetune_ner():
    # basic conf
    _config = {
        "write": temp_log_file_location(),
        "port": 5022,
        "fine_tune_spacy_ner": True,
        "backend": "spacy_sklearn",
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None

    entities = trained.entity_extractor.extract_entities(trained.nlp, u"I am living in New York now.")
    # Although the model is trained on restaurant entities, we can use the entities (`GPE`, `DATE`)
    # from spacy since we are fine tuning. This should even be the case if the rasa-entity training data changes!
    assert {u'start': 15, u'end': 23, u'value': u'New York', u'entity': u'GPE'} in entities
    assert {u'start': 24, u'end': 27, u'value': u'now', u'entity': u'DATE'} in entities


def test_train_spacy_sklearn_multithread():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": SPACY_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "num_threads": 2,
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    trained = run_train(_config)
    assert trained.entity_extractor is not None
    assert trained.intent_classifier is not None
