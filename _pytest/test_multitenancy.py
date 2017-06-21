# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import os
import signal
import tempfile
import requests

import pytest
import sys
from multiprocessing import Semaphore

import time

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.server import RasaNLU
from utilities import ResponseTest


@pytest.fixture(scope="module")
def stub(component_builder):
    sem = Semaphore(1)
    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "backend": "mitie",
        "path": os.path.join(root_dir, "test_models"),
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "server_model_dirs": {
            "one": "test_model_mitie",
            "two": "test_model_mitie_sklearn",
            "three": "test_model_spacy_sklearn",
        }
    }
    url = '127.0.0.1'
    port = 5000
    pid = os.fork()
    if pid == 0:
        sem.acquire()
        config = RasaNLUConfig(cmdline_args=_config)
        rasa = RasaNLU(config, component_builder)
        sem.release()
        rasa.app.run(url, port)
        rasa.data_router.__del__()
        os._exit(0)
    else:
        time.sleep(3)
        sem.acquire()
        sem.release()

    def with_base_url(method):
        """
        Save some typing and ensure we always use our own pool.
        """

        def request(path, *args, **kwargs):
            return method("http://{}:{}".format(url, port) + path, *args, **kwargs)

        return request

    yield with_base_url

    if pid != 0:
        os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food&model=one",
        {"entities": [], "intent": "affirm", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food&model=two",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food&model=three",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
])
def test_get_parse(stub, response_test):
    response = stub(requests.get)(response_test.endpoint)
    rjs = response.json()

    assert response.status_code == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food",
        {"error": "No model found with alias 'default'. Error: Failed to load model metadata. "}
    ),
    ResponseTest(
        "/parse?q=food&model=umpalumpa",
        {"error": "No model found with alias 'umpalumpa'. Error: Failed to load model metadata. "}
    )
])
def test_get_parse_invalid_model(stub, response_test):
    response = stub(requests.get)(response_test.endpoint)
    rjs = response.json()
    assert response.status_code == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "affirm", "text": "food"},
        payload={"q": "food", "model": "one"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "model": "two"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "model": "three"}
    ),
])
def test_post_parse(stub, response_test):
    response = stub(requests.post)(response_test.endpoint, json=response_test.payload)
    rjs = response.json()
    assert response.status_code == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"error": "No model found with alias 'default'. Error: Failed to load model metadata. "},
        payload={"q": "food"}
    ),
    ResponseTest(
        "/parse",
        {"error": "No model found with alias 'umpalumpa'. Error: Failed to load model metadata. "},
        payload={"q": "food", "model": "umpalumpa"}
    ),
])
def test_post_parse_invalid_model(stub, response_test):
    response = stub(requests.post)(response_test.endpoint, json=response_test.payload)
    rjs = response.json()
    assert response.status_code == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


if __name__ == '__main__':
    # Retrain different multitenancy models
    def train(cfg_name, model_name):
        from rasa_nlu.train import create_persistor
        from rasa_nlu.converters import load_data

        config = RasaNLUConfig(cfg_name)
        trainer = Trainer(config)
        training_data = load_data(config['data'])

        trainer.train(training_data)
        persistor = create_persistor(config)
        trainer.persist("test_models", persistor, model_name=model_name)

    train("config_mitie.json", "test_model_mitie")
    train("config_spacy.json", "test_model_spacy_sklearn")
    train("config_mitie_sklearn.json", "test_model_mitie_sklearn")
