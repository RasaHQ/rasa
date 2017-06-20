# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import tempfile

import pytest
import time

import utilities
from rasa_nlu.config import RasaNLUConfig
import json
import io
from treq.testing import StubTreq

from utilities import ResponseTest
from rasa_nlu.server import RasaNLU


@pytest.fixture(scope="module")
def app(tmpdir_factory):
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
    return StubTreq(RasaNLU(config).app.resource())


@pytest.fixture
def rasa_default_train_data():
    with io.open('data/examples/rasa/demo-rasa.json',
                 encoding='utf-8') as train_file:
        return json.loads(train_file.read())


@pytest.inlineCallbacks
def test_root(app):
    response = yield app.get("http://localhost:5000/")
    assert response.status_code == 200 and response.text.startswith(b"hello")


@pytest.inlineCallbacks
def test_status(app):
    response = yield app.get("http://localhost:5000/status")
    rjs = response.json()
    assert response.status_code == 200 and \
           ("trainings_under_this_process" in rjs and "available_models" in rjs)


@pytest.inlineCallbacks
def test_config(app):
    response = yield app.get("http://localhost:5000/config")
    assert response.status_code == 200


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://localhost:5000/version")
    rjs = response.json()
    assert response.status_code == 200 and \
           ("version" in rjs)


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://localhost:5000/parse?q=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}]
    ),
    ResponseTest(
        "http://localhost:5000/parse?q=hello ńöñàśçií",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}]
    ),
    ResponseTest(
        "http://localhost:5000/parse?q=",
        [{"entities": {}, "confidence": 0.0, "intent": None, "_text": ""}]
    ),
])
@pytest.inlineCallbacks
def test_get_parse(app, response_test):
    response = yield app.get(response_test.endpoint)
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert all(prop in response.json()[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://localhost:5000/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}],
        payload={"q": "hello"}
    ),
    ResponseTest(
        "http://localhost:5000/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}],
        payload={"q": "hello ńöñàśçií"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint,
                              data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert all(prop in response.json()[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
@pytest.inlineCallbacks
def test_post_train(app, rasa_default_train_data):
    response = yield app.post("http://localhost:5000/train", data=json.dumps(rasa_default_train_data),
                              content_type='application/json')
    assert response.status_code == 200
    assert len(response.json()["training_process_ids"]) == 1
    assert response.json()["info"] == "training started."


@pytest.inlineCallbacks
def test_model_hot_reloading(app, rasa_default_train_data):
    query = "http://localhost:5000/parse?q=hello&model=my_keyword_model"
    response = yield app.get(query)
    assert response.status_code == 404, "Model should not exist yet"
    response = yield app.post("http://localhost:5000/train?name=my_keyword_model&pipeline=keyword",
                              data=json.dumps(rasa_default_train_data),
                              content_type='application/json')
    assert response.status_code == 200, "Training should start successfully"
    time.sleep(3)  # training should be quick as the keyword model doesn't do any training
    response = yield app.get(query)
    assert response.status_code == 200, "Model should now exist after it got trained"
