# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tempfile

import pytest
import time

from treq.testing import StubTreq

from rasa_nlu.config import RasaNLUConfig
import json
import io

import utilities
from utilities import ResponseTest
from rasa_nlu.server import RasaNLU


@pytest.fixture(scope="module")
def app(tmpdir_factory):
    """
    This fixture makes use of the IResource interface of the Klein application to mock Rasa HTTP server.
    :param component_builder:
    :return:
    """

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "pipeline": "keyword",
        "path": tmpdir_factory.mktemp("models").strpath,
        "server_model_dirs": {},
        "data": "./data/demo-restaurants.json",
        "emulate": "wit",
        "max_training_processes": 1
    }

    config = RasaNLUConfig(cmdline_args=_config)
    rasa = RasaNLU(config, testing=True)
    return StubTreq(rasa.app.resource())


@pytest.fixture
def rasa_default_train_data():
    with io.open('data/examples/rasa/demo-rasa.json',
                 encoding='utf-8') as train_file:
        return json.loads(train_file.read())


@pytest.inlineCallbacks
def test_root(app):
    response = yield app.get("http://dummy_uri/")
    content = yield response.text()
    assert response.code == 200 and content.startswith("hello")


@pytest.inlineCallbacks
def test_status(app):
    response = yield app.get("http://dummy_uri/status")
    rjs = yield response.json()
    assert response.code == 200 and ('trainings_queued' in rjs and 'training_workers' in rjs)
    assert rjs['training_workers'] == 1


@pytest.inlineCallbacks
def test_config(app):
    response = yield app.get("http://dummy_uri/config")
    assert response.code == 200


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy_uri/version")
    rjs = yield response.json()
    assert response.code == 200 and ("version" in rjs)


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy_uri/parse?q=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}]
    ),
    ResponseTest(
        "http://dummy_uri/parse?query=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}]
    ),
    ResponseTest(
        "http://dummy_uri/parse?q=hello ńöñàśçií",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}]
    ),
    ResponseTest(
        "http://dummy_uri/parse?q=",
        [{"entities": {}, "confidence": 0.0, "intent": None, "_text": ""}]
    ),
])
@pytest.inlineCallbacks
def test_get_parse(app, response_test):
    response = yield app.get(response_test.endpoint)
    rjs = yield response.json()
    assert response.code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy_uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}],
        payload={"q": "hello"}
    ),
    ResponseTest(
        "http://dummy_uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello"}],
        payload={"query": "hello"}
    ),
    ResponseTest(
        "http://dummy_uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet", "_text": "hello ńöñàśçií"}],
        payload={"q": "hello ńöñàśçií"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint, data=json.dumps(response_test.payload),
                              content_type='application/json')
    rjs = yield response.json()
    assert response.code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
@pytest.inlineCallbacks
def test_post_train(app, rasa_default_train_data):
    response = app.post("http://dummy_uri/train", data=json.dumps(rasa_default_train_data),
                        content_type='application/json')
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 200
    assert rjs["info"].startswith("new model trained: ")


@pytest.inlineCallbacks
def test_model_hot_reloading(app, rasa_default_train_data):
    query = "http://dummy_uri/parse?q=hello&model=my_keyword_model"
    response = yield app.get(query)
    assert response.code == 404, "Model should not exist yet"

    response = app.post("http://dummy_uri/train?name=my_keyword_model&pipeline=keyword",
                        data=json.dumps(rasa_default_train_data), content_type='application/json')
    time.sleep(3)
    app.flush()
    response = yield response
    assert response.code == 200, "Training should end successfully"

    response = yield app.get(query)
    assert response.code == 200, "Model should now exist after it got trained"
