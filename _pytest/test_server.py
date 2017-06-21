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
import time

import utilities
from rasa_nlu.config import RasaNLUConfig
import json
import io

from utilities import ResponseTest
from rasa_nlu.server import RasaNLU


@pytest.fixture(scope="module")
def stub(tmpdir_factory):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "backend": "mitie",
        "path": tmpdir_factory.mktemp("models").strpath,
        "server_model_dirs": {},
        "data": "./data/demo-restaurants.json",
        "emulate": "wit",
    }
    config = RasaNLUConfig(cmdline_args=_config)
    url = '127.0.0.1'
    port = 5000
    rasa = RasaNLU(config)
    pid = os.fork()
    if pid == 0:
        rasa.app.run(url, port)
        os._exit(0)

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


@pytest.fixture
def rasa_default_train_data():
    with io.open('data/examples/rasa/demo-rasa.json',
                 encoding='utf-8') as train_file:
        return json.loads(train_file.read())


def test_root(stub):
    response = stub(requests.get)("/")
    content = response.content
    assert response.status_code == 200 and content.startswith(b"hello")


def test_status(stub):
    response = stub(requests.get)("/status")
    rjs = response.json()
    assert response.status_code == 200 and ("trainings_under_this_process" in rjs and "available_models" in rjs)


def test_config(stub):
    response = stub(requests.get)("/config")
    assert response.status_code == 200


def test_version(stub):
    response = stub(requests.get)("/version")
    rjs = response.json()
    assert response.status_code == 200 and ("version" in rjs)


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}]
    ),
    ResponseTest(
        "/parse?q=hello ńöñàśçií",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}]
    ),
    ResponseTest(
        "/parse?q=",
        [{"entities": {}, "confidence": 0.0, "intent": None, "_text": ""}]
    ),
])
def test_get_parse(stub, response_test):
    response = stub(requests.get)(response_test.endpoint)
    rjs = response.json()
    assert response.status_code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}],
        payload={"q": "hello"}
    ),
    ResponseTest(
        "/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}],
        payload={"q": "hello ńöñàśçií"}
    ),
])
def test_post_parse(stub, response_test):
    response = stub(requests.post)(response_test.endpoint, json=response_test.payload)
    rjs = response.json()
    assert response.status_code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
def test_post_train(stub, rasa_default_train_data):
    response = stub(requests.post)("/train", json=rasa_default_train_data)
    rjs = response.json()
    assert response.status_code == 200
    assert len(rjs["training_process_ids"]) == 0
    assert rjs["info"] == "training started."


def test_model_hot_reloading(stub, rasa_default_train_data):
    query = "/parse?q=hello&model=my_keyword_model"
    response = stub(requests.get)(query)
    assert response.status_code == 404, "Model should not exist yet"
    response = stub(requests.post)("/train?name=my_keyword_model&pipeline=keyword", json=rasa_default_train_data)
    assert response.status_code == 200, "Training should start successfully"
    time.sleep(3)  # training should be quick as the keyword model doesn't do any training
    response = stub(requests.get)(query)
    assert response.status_code == 200, "Model should now exist after it got trained"
