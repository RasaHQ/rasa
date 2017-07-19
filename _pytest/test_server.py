# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import signal
import tempfile

import requests

import pytest
import time
from multiprocessing import Semaphore

from rasa_nlu.config import RasaNLUConfig
import json
import io

import utilities
from utilities import ResponseTest
from rasa_nlu.server import RasaNLU


@pytest.fixture(scope="module")
def http_test_server(tmpdir_factory):
    """
    Launches a Rasa HTTP server instance on a subprocess for testing.
    This is necessary as Klein HTTP application's endpoints cannot by tested directly.
    The twisted/treq library could do that but it is not compatible with pytest
    and pytest's plugins related to treq are no more maintained.
    """
    sem = Semaphore(1)
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
    url = '127.0.0.1'
    port = 5000
    pid = os.fork()
    if pid == 0:
        sem.acquire()

        config = RasaNLUConfig(cmdline_args=_config)
        rasa = RasaNLU(config)
        sem.release()
        rasa.app.run(url, port)
        rasa.data_router.shutdown()
        os._exit(0)

    else:
        time.sleep(3)
        sem.acquire()
        sem.release()

    def with_base_url(method):
        """Save some typing and ensure we always use our own pool."""

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


def test_root(http_test_server):
    response = http_test_server(requests.get)("/")
    content = response.content
    assert response.status_code == 200 and content.startswith(b"hello")


def test_status(http_test_server):
    response = http_test_server(requests.get)("/status")
    rjs = response.json()
    assert response.status_code == 200 and ('training_queued' in rjs and 'training_workers' in rjs)
    assert rjs['training_workers'] > 0


def test_config(http_test_server):
    response = http_test_server(requests.get)("/config")
    assert response.status_code == 200


def test_version(http_test_server):
    response = http_test_server(requests.get)("/version")
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
def test_get_parse(http_test_server, response_test):
    response = http_test_server(requests.get)(response_test.endpoint)
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
def test_post_parse(http_test_server, response_test):
    response = http_test_server(requests.post)(response_test.endpoint, json=response_test.payload)
    rjs = response.json()
    assert response.status_code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
def test_post_train(http_test_server, rasa_default_train_data):
    response = http_test_server(requests.post)("/train", json=rasa_default_train_data)
    rjs = response.json()
    assert response.status_code == 200
    assert rjs["info"] == "training started."


def test_model_hot_reloading(http_test_server, rasa_default_train_data):
    query = "/parse?q=hello&model=my_keyword_model"
    response = http_test_server(requests.get)(query)
    assert response.status_code == 404, "Model should not exist yet"
    response = http_test_server(requests.post)("/train?name=my_keyword_model&pipeline=keyword",
                                               json=rasa_default_train_data)
    assert response.status_code == 200, "Training should start successfully"
    time.sleep(5)  # training should be quick as the keyword model doesn't do any training
    response = http_test_server(requests.get)(query)
    assert response.status_code == 200, "Model should now exist after it got trained"
