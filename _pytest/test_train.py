# -*- coding: utf-8 -*-
import tempfile

import pytest
import os
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.train import do_train
from rasa_nlu.utils.mitie import MITIE_BACKEND_NAME
from rasa_nlu.utils.spacy import SPACY_BACKEND_NAME


def run_train(_config):
    config = RasaNLUConfig(cmdline_args=_config)
    do_train(config)


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
    run_train(_config)


def test_train_mitie_sklearn():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": "mitie_sklearn",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)


def test_train_mitie_noents():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": MITIE_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    run_train(_config)


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
    run_train(_config)


def test_train_spacy_sklearn():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": SPACY_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)


def test_train_spacy_sklearn_noents():
    # basic conf
    _config = {
        'write': temp_log_file_location(),
        'port': 5022,
        "backend": SPACY_BACKEND_NAME,
        "path": tempfile.mkdtemp(),
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    run_train(_config)


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
    run_train(_config)
