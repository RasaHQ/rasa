# -*- coding: utf-8 -*-

import pytest
import os
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.train import do_train


def run_train(_config):
    config = RasaNLUConfig(cmdline_args=_config)
    do_train(config)


def test_train_mitie():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "mitie",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)


def test_train_mitie_noents():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "mitie",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    run_train(_config)


def test_train_mitie_multithread():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "mitie",
        "path": "./",
        "num_threads": 2,
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)


def test_train_spacy_sklearn():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "spacy_sklearn",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)


def test_train_spacy_sklearn_noents():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "spacy_sklearn",
        "path": "./",
        "data": "./data/examples/rasa/demo-rasa-noents.json"
    }
    run_train(_config)


def test_train_spacy_sklearn_multithread():
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "spacy_sklearn",
        "path": "./",
        "num_threads": 2,
        "data": "./data/examples/rasa/demo-rasa.json"
    }
    run_train(_config)
