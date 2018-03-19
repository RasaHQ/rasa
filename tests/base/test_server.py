# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import tempfile
import time

import pytest
from treq.testing import StubTreq

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.server import RasaNLU
from tests import utilities
from tests.utilities import ResponseTest


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
        "path": tmpdir_factory.mktemp("projects").strpath,
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
                 encoding='utf-8-sig') as train_file:
        return json.loads(train_file.read())


@pytest.inlineCallbacks
def test_root(app):
    response = yield app.get("http://dummy-uri/")
    content = yield response.text()
    assert response.code == 200 and content.startswith("hello")


@pytest.inlineCallbacks
def test_status(app):
    response = yield app.get("http://dummy-uri/status")
    rjs = yield response.json()
    assert response.code == 200 and "available_projects" in rjs
    assert "default" in rjs["available_projects"]


@pytest.inlineCallbacks
def test_config(app):
    response = yield app.get("http://dummy-uri/config")
    assert response.code == 200


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy-uri/version")
    rjs = yield response.json()
    assert response.code == 200 and "version" in rjs


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy-uri/parse?q=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello"}]
    ),
    ResponseTest(
        "http://dummy-uri/parse?query=hello",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello"}]
    ),
    ResponseTest(
        "http://dummy-uri/parse?q=hello ńöñàśçií",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello ńöñàśçií"}]
    ),
    ResponseTest(
        "http://dummy-uri/parse?q=",
        [{"entities": {}, "confidence": 0.0, "intent": None, "_text": ""}]
    ),
])
@pytest.inlineCallbacks
def test_get_parse(app, response_test):
    response = yield app.get(response_test.endpoint)
    rjs = yield response.json()
    assert response.code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in
               ['entities', 'intent', '_text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy-uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello"}],
        payload={"q": "hello"}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello"}],
        payload={"query": "hello"}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        [{"entities": {}, "confidence": 1.0, "intent": "greet",
          "_text": "hello ńöñàśçií"}],
        payload={"q": "hello ńöñàśçií"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint,
                              json=response_test.payload)
    rjs = yield response.json()
    assert response.code == 200
    assert len(rjs) == 1
    assert all(prop in rjs[0] for prop in
               ['entities', 'intent', '_text', 'confidence'])


@utilities.slowtest
@pytest.inlineCallbacks
def test_post_train(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/train", json=rasa_default_train_data)
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 404, "A project name to train must be specified"
    assert "error" in rjs


@utilities.slowtest
@pytest.inlineCallbacks
def test_post_train_internal_error(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/train?project=test",
                        json={"data": "dummy_data_for_triggering_an_error"})
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 500, "The training data format is not valid"
    assert "error" in rjs


@pytest.inlineCallbacks
def test_model_hot_reloading(app, rasa_default_train_data):
    query = "http://dummy-uri/parse?q=hello&project=my_keyword_model"
    response = yield app.get(query)
    assert response.code == 404, "Project should not exist yet"
    train_u = "http://dummy-uri/train?project=my_keyword_model&pipeline=keyword"
    response = app.post(train_u, json=rasa_default_train_data)
    time.sleep(3)
    app.flush()
    response = yield response
    assert response.code == 200, "Training should end successfully"
    response = yield app.get(query)
    assert response.code == 200, "Project should now exist after it got trained"


@pytest.inlineCallbacks
def test_evaluate_invalid_project_error(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/evaluate",
                        json=rasa_default_train_data,
                        params={"project": "project123"})
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 500, "The project cannot be found"
    assert "error" in rjs
    assert rjs["error"] == "Project project123 could not be found"


@pytest.inlineCallbacks
def test_evaluate_internal_error(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/evaluate",
                        json={"data": "dummy_data_for_triggering_an_error"})
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 500, "The training data format is not valid"
    assert "error" in rjs
    assert "Unknown data format for file" in rjs["error"]


@pytest.inlineCallbacks
def test_evaluate(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/evaluate",
                        json=rasa_default_train_data)
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 200, "Evaluation should start"
    assert "intent_evaluation" in rjs
    assert all(prop in rjs["intent_evaluation"] for prop in ["report",
                                                             "predictions",
                                                             "precision",
                                                             "f1_score",
                                                             "accuracy"])
