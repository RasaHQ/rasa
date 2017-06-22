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

from utilities import ResponseTest
from rasa_nlu.server import create_app


@pytest.fixture(scope="module")
def app(tmpdir_factory):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,                 # unused in test app
        "backend": "mitie",
        "path": tmpdir_factory.mktemp("models").strpath,
        "server_model_dirs": {},
        "data": "./data/demo-restaurants.json",
        "emulate": "wit",
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = create_app(config)
    return application


@pytest.fixture
def rasa_default_train_data():
    with io.open('data/examples/rasa/demo-rasa.json',
                 encoding='utf-8') as train_file:
        return json.loads(train_file.read())


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200 and response.data.startswith(b"hello")


def test_status(client):
    response = client.get("/status")
    rjs = response.json
    assert response.status_code == 200 and \
        ("trainings_under_this_process" in rjs and "available_models" in rjs)


def test_config(client):
    response = client.get("/config")
    assert response.status_code == 200


def test_version(client):
    response = client.get("/version")
    rjs = response.json
    assert response.status_code == 200 and \
        ("version" in rjs)


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
def test_get_parse(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 200
    assert len(response.json) == 1
    assert all(prop in response.json[0] for prop in ['entities', 'intent', '_text', 'confidence'])


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
def test_post_parse(client, response_test):
    response = client.post(response_test.endpoint,
                           data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 200
    assert len(response.json) == 1
    assert all(prop in response.json[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
def test_post_train(client, rasa_default_train_data):
    response = client.post("/train", data=json.dumps(rasa_default_train_data), content_type='application/json')
    assert response.status_code == 200
    assert len(response.json["training_process_ids"]) == 1
    assert response.json["info"] == "training started."


def test_model_hot_reloading(client, rasa_default_train_data):
    query = "/parse?q=hello&model=my_keyword_model"
    response = client.get(query)
    assert response.status_code == 404, "Model should not exist yet"
    response = client.post("/train?name=my_keyword_model&pipeline=keyword",
                           data=json.dumps(rasa_default_train_data),
                           content_type='application/json')
    assert response.status_code == 200, "Training should start successfully"
    time.sleep(3)    # training should be quick as the keyword model doesn't do any training
    response = client.get(query)
    assert response.status_code == 200, "Model should now exist after it got trained"


def test_wsgi():
    # this avoids the loading of any models when starting the server --> faster
    os.environ["RASA_path"] = "some_none/existent/path"
    from rasa_nlu.wsgi import application
    assert application is not None
    del os.environ["RASA_path"]
