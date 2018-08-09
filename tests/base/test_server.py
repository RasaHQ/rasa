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
import yaml
from treq.testing import StubTreq

from rasa_nlu import utils
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.data_router import DataRouter
from rasa_nlu.server import RasaNLU
from tests import utilities
from tests.utilities import ResponseTest


@pytest.fixture(scope="module")
def app(tmpdir_factory):
    """Use IResource interface of Klein to mock Rasa HTTP server.

    :param component_builder:
    :return:
    """

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    router = DataRouter(tmpdir_factory.mktemp("projects").strpath)
    rasa = RasaNLU(router,
                   logfile=nlu_log_file,
                   testing=True)
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
    assert "current_training_processes" in rjs
    assert "max_training_processes" in rjs
    assert "default" in rjs["available_projects"]


@pytest.inlineCallbacks
def test_config(app):
    response = yield app.get("http://dummy-uri/config")
    assert response.code == 200


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy-uri/version")
    rjs = yield response.json()
    assert response.code == 200
    assert set(rjs.keys()) == {"version", "minimum_compatible_version"}


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy-uri/parse?q=hello",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'}, 'text': 'hello'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?query=hello",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'}, 'text': 'hello'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?q=hello ńöñàśçií",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'},
         'text': 'hello ńöñàśçií'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?q=",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 0.0, 'name': None}, 'text': ''}
    ),
])
@pytest.inlineCallbacks
def test_get_parse(app, response_test):
    response = yield app.get(response_test.endpoint)
    rjs = yield response.json()
    assert response.code == 200
    assert rjs == response_test.expected_response
    assert all(prop in rjs for prop in
               ['project', 'entities', 'intent',
                'text', 'model'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy-uri/parse",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'},
         'text': 'hello'},
        payload={"q": "hello"}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'},
         'text': 'hello'},
        payload={"query": "hello"}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        {'project': 'default', 'entities': [], 'model': 'fallback',
         'intent': {'confidence': 1.0, 'name': 'greet'},
         'text': 'hello ńöñàśçií'},
        payload={"q": "hello ńöñàśçií"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint,
                              json=response_test.payload)
    rjs = yield response.json()
    assert response.code == 200
    assert rjs == response_test.expected_response
    assert all(prop in rjs for prop in
               ['project', 'entities', 'intent',
                'text', 'model'])


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
    train_u = "http://dummy-uri/train?project=my_keyword_model"
    model_config = {"pipeline": "keyword", "data": rasa_default_train_data}
    model_str = yaml.safe_dump(model_config, default_flow_style=False,
                               allow_unicode=True)
    response = app.post(train_u,
                        headers={b"Content-Type": b"application/x-yml"},
                        data=model_str)
    time.sleep(3)
    app.flush()
    response = yield response
    assert response.code == 200, "Training should end successfully"

    response = app.post(train_u,
                        headers={b"Content-Type": b"application/json"},
                        data=json.dumps(model_config))
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


@pytest.inlineCallbacks
def test_unload_model_error(app):
    project_err = "http://dummy-uri/models?project=my_project&model=my_model"
    response = yield app.delete(project_err)
    rjs = yield response.json()
    assert response.code == 500, "Project not found"
    assert rjs['error'] == "Project my_project could not be found"

    model_err = "http://dummy-uri/models?model=my_model"
    response = yield app.delete(model_err)
    rjs = yield response.json()
    assert response.code == 500, "Model not found"
    assert rjs['error'] == "Failed to unload model my_model for project default."


@pytest.inlineCallbacks
def test_unload_fallback(app):
    unload = "http://dummy-uri/models?model=fallback"
    response = yield app.delete(unload)
    rjs = yield response.json()
    assert response.code == 200, "Fallback model unloaded"
    assert rjs == "fallback"
