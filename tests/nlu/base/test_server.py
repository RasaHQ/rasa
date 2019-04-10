# -*- coding: utf-8 -*-
import asyncio
import io
import os
import json
import tempfile
import shutil
import time

import pytest
import ruamel.yaml as yaml
from treq.testing import StubTreq

from rasa.nlu.data_router import DataRouter
from rasa.nlu.server import RasaNLU
from tests.nlu import utilities
from tests.nlu.utilities import ResponseTest

KEYWORD_MODEL_NAME = 'keywordmodel'

KEYWORD_PROJECT_NAME = 'keywordproject'


@pytest.fixture(scope="module")
def app(tmpdir_factory, trained_nlu_model):
    """Use IResource interface of Klein to mock Rasa HTTP server"""

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    temp_path = tmpdir_factory.mktemp("projects")
    try:
        shutil.copytree(trained_nlu_model,
                        os.path.join(temp_path.strpath,
                                     "{}/{}".format(KEYWORD_PROJECT_NAME,
                                                    KEYWORD_MODEL_NAME)))
    except FileExistsError:
        pass

    router = DataRouter(temp_path.strpath)
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
    assert "current_pool_processes" in rjs
    assert "max_pool_processes" in rjs
    assert KEYWORD_PROJECT_NAME in rjs["available_projects"]


@pytest.inlineCallbacks
def test_version(app):
    response = yield app.get("http://dummy-uri/version")
    rjs = yield response.json()
    assert response.code == 200
    assert set(rjs.keys()) == {"version", "minimum_compatible_version"}


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy-uri/parse?project={}&model={}"
        "&q=hello".format(KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME),
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?project={}&model={}"
        "&query=hello".format(KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME),
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?project={}&model={}"
        "&q=hello ńöñàśçií".format(KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME),
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello ńöñàśçií'}
    ),
    ResponseTest(
        "http://dummy-uri/parse?project={}&model={}"
        "&q=".format(KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME),
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 0.0,
                                                 'name': None}, 'text': ''}
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
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello'},
        payload={"q": "hello",
                 "project": KEYWORD_PROJECT_NAME,
                 "model": KEYWORD_MODEL_NAME}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello'},
        payload={"query": "hello",
                 "project": KEYWORD_PROJECT_NAME,
                 "model": KEYWORD_MODEL_NAME}
    ),
    ResponseTest(
        "http://dummy-uri/parse",
        {'project': KEYWORD_PROJECT_NAME, 'entities': [],
         'model': KEYWORD_MODEL_NAME, 'intent': {'confidence': 1.0,
                                                 'name': 'greet'},
         'text': 'hello ńöñàśçií'},
        payload={"q": "hello ńöñàśçií",
                 "project": KEYWORD_PROJECT_NAME,
                 "model": KEYWORD_MODEL_NAME}
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
    response = app.post("http://dummy-uri/train",
                        json=rasa_default_train_data)
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 404, "A project name to train must be specified"
    assert "error" in rjs


@utilities.slowtest
@pytest.inlineCallbacks
def test_post_train_success(app, rasa_default_train_data):
    import zipfile
    model_config = {"pipeline": "keyword", "data": rasa_default_train_data}

    response = app.post("http://dummy-uri/train?project=test&model=test",
                        json=model_config)
    time.sleep(3)
    app.flush()
    response = yield response
    content = yield response.content()
    assert response.code == 200
    assert zipfile.ZipFile(io.BytesIO(content)).testzip() is None


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
    assert response.code == 200, ("Project should now exist "
                                  "after it got trained")


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
    assert rjs["error"] == "Project 'project123' could not be found."


@pytest.inlineCallbacks
def test_evaluate_no_model(app):
    response = app.post("http://dummy-uri/evaluate")
    time.sleep(2)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 500, "No model in project 'default' to evaluate"
    assert "error" in rjs


@pytest.inlineCallbacks
def test_evaluate_internal_error(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/evaluate",
                        json={"data": "dummy_data_for_triggering_an_error"})
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 500, "The training data format is not valid"


@pytest.inlineCallbacks
def test_unload_model_error(app):
    FAKE_PROJECT = "fakeproject"
    NON_EXISTING_MODEL_NAME = "fakemodel"

    project_err = ("http://dummy-uri/models"
                   "?project={}&model={}".format(FAKE_PROJECT,
                                                 NON_EXISTING_MODEL_NAME))
    response = yield app.delete(project_err)
    rjs = yield response.json()
    assert response.code == 500, "Project not found"
    assert rjs['error'] == ("Project {} could " 
                            "not be found".format(FAKE_PROJECT))

    model_err = ("http://dummy-uri/models?project={}"
                 "&model={}".format(KEYWORD_PROJECT_NAME,
                                    NON_EXISTING_MODEL_NAME))
    response = yield app.delete(model_err)
    rjs = yield response.json()
    assert response.code == 500, "Model not found"
    assert rjs['error'] == ("Failed to unload model {} for project "
                            "{}.".format(NON_EXISTING_MODEL_NAME,
                                         KEYWORD_PROJECT_NAME))


@pytest.inlineCallbacks
def test_unload(app):
    unload = ("http://dummy-uri/models?project={}"
              "&model={}".format(KEYWORD_PROJECT_NAME,
                                 KEYWORD_MODEL_NAME))
    response = yield app.delete(unload)
    rjs = yield response.json()
    assert response.code == 200, "Fallback model unloaded"
    assert rjs == KEYWORD_MODEL_NAME


@pytest.inlineCallbacks
def test_evaluate(app, rasa_default_train_data):
    response = app.post("http://dummy-uri/evaluate"
                        "?project={}&model={}".format(KEYWORD_PROJECT_NAME,
                                                      KEYWORD_MODEL_NAME),
                        json=rasa_default_train_data)
    time.sleep(3)
    app.flush()
    response = yield response
    rjs = yield response.json()
    assert response.code == 200, "Evaluation should start"
    assert "intent_evaluation" in rjs
    assert "entity_evaluation" in rjs
    assert all(prop in rjs["intent_evaluation"] for prop in ["report",
                                                             "predictions",
                                                             "precision",
                                                             "f1_score",
                                                             "accuracy"])
