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
import tempfile


from rasa.nlu.data_router import DataRouter
from rasa.nlu.server import create_app
from tests.nlu import utilities
from tests.nlu.utilities import ResponseTest


KEYWORD_PROJECT_NAME = "keywordproject"
KEYWORD_MODEL_NAME = "keywordmodel"


@pytest.fixture
def app(tmpdir_factory, trained_nlu_model):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    temp_path = tmpdir_factory.mktemp("projects")
    try:
        shutil.copytree(
            trained_nlu_model,
            os.path.join(
                temp_path.strpath,
                "{}/{}".format(KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME),
            ),
        )
    except FileExistsError:
        pass

    router = DataRouter(temp_path.strpath)

    rasa = create_app(router, logfile=nlu_log_file)

    return rasa.test_client


@pytest.fixture
def rasa_default_train_data():
    with io.open(
        "data/examples/rasa/demo-rasa.json", encoding="utf-8-sig"
    ) as train_file:
        return json.loads(train_file.read())


def test_root(app):
    _, response = app.get("/")
    content = response.text
    assert response.status == 200 and content.startswith("Hello")


def test_status(app):
    _, response = app.get("/status")
    rjs = response.json
    assert response.status == 200 and "available_projects" in rjs
    assert "current_worker_processes" in rjs
    assert "max_worker_processes" in rjs
    assert KEYWORD_PROJECT_NAME in rjs["available_projects"]


def test_version(app):
    _, response = app.get("/version")
    rjs = response.json
    assert response.status == 200
    assert set(rjs.keys()) == {"version", "minimum_compatible_version"}


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=hello&model={}&project={}".format(
                KEYWORD_MODEL_NAME, KEYWORD_PROJECT_NAME
            ),
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello&model={}&project={}".format(
                KEYWORD_MODEL_NAME, KEYWORD_PROJECT_NAME
            ),
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello ńöñàśçií&model={}&project={}".format(
                KEYWORD_MODEL_NAME, KEYWORD_PROJECT_NAME
            ),
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
        ),
        ResponseTest(
            "/parse?q=&model={}&project={}".format(
                KEYWORD_MODEL_NAME, KEYWORD_PROJECT_NAME
            ),
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 0.0, "name": None},
                "text": "",
            },
        ),
    ],
)
def test_get_parse(app, response_test):
    _, response = app.get(response_test.endpoint)
    rjs = response.json
    assert response.status == 200
    assert rjs == response_test.expected_response
    assert all(
        prop in rjs for prop in ["project", "entities", "intent", "text", "model"]
    )


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse",
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={
                "q": "hello",
                "project": KEYWORD_PROJECT_NAME,
                "model": KEYWORD_MODEL_NAME,
            },
        ),
        ResponseTest(
            "/parse",
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={
                "query": "hello",
                "project": KEYWORD_PROJECT_NAME,
                "model": KEYWORD_MODEL_NAME,
            },
        ),
        ResponseTest(
            "/parse",
            {
                "project": KEYWORD_PROJECT_NAME,
                "entities": [],
                "model": KEYWORD_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={
                "q": "hello ńöñàśçií",
                "project": KEYWORD_PROJECT_NAME,
                "model": KEYWORD_MODEL_NAME,
            },
        ),
    ],
)
def test_post_parse(app, response_test):
    _, response = app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert rjs == response_test.expected_response
    assert all(
        prop in rjs for prop in ["project", "entities", "intent", "text", "model"]
    )


@utilities.slowtest
def test_post_train(app, rasa_default_train_data):
    _, response = app.post("/train", json=rasa_default_train_data)
    rjs = response.json
    assert response.status == 404, "A project name to train must be specified"
    assert "details" in rjs


@utilities.slowtest
def test_post_train_success(app, rasa_default_train_data):
    import zipfile

    model_config = {"pipeline": "keyword", "data": rasa_default_train_data}

    _, response = app.post("/train?project=test&model=test", json=model_config)

    content = response.body
    assert response.status == 200
    assert zipfile.ZipFile(io.BytesIO(content)).testzip() is None


@utilities.slowtest
def test_post_train_internal_error(app, rasa_default_train_data):
    _, response = app.post(
        "/train?project=test", json={"data": "dummy_data_for_triggering_an_error"}
    )
    rjs = response.json
    assert response.status == 500, "The training data format is not valid"


def test_model_hot_reloading(app, rasa_default_train_data):
    query = "/parse?q=hello&project=my_keyword_model"
    _, response = app.get(query)
    assert response.status == 404, "Project should not exist yet"
    train_u = "/train?project=my_keyword_model"
    model_config = {"pipeline": "keyword", "data": rasa_default_train_data}
    model_str = yaml.safe_dump(
        model_config, default_flow_style=False, allow_unicode=True
    )
    _, response = app.post(
        train_u, headers={"Content-Type": "application/x-yml"}, data=model_str
    )
    assert response.status == 200, "Training should end successfully"

    _, response = app.post(
        train_u,
        headers={"Content-Type": "application/json"},
        data=json.dumps(model_config),
    )
    assert response.status == 200, "Training should end successfully"

    _, response = app.get(query)
    assert response.status == 200, "Project should now exist after it got trained"


def test_evaluate_invalid_project_error(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate", json=rasa_default_train_data, params={"project": "project123"}
    )

    rjs = response.json
    assert response.status == 500, "The project cannot be found"
    assert "details" in rjs
    assert rjs["details"]["error"] == "Project 'project123' could not be found."


def test_evaluate_internal_error(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate", json={"data": "dummy_data_for_triggering_an_error"}
    )

    assert response.status == 500, "The training data format is not valid"


def test_evaluate(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate?project={}&model={}".format(
            KEYWORD_PROJECT_NAME, KEYWORD_MODEL_NAME
        ),
        json=rasa_default_train_data,
    )

    rjs = response.json
    assert response.status == 200, "Evaluation should start"
    assert "intent_evaluation" in rjs
    assert "entity_evaluation" in rjs
    assert all(
        prop in rjs["intent_evaluation"]
        for prop in ["report", "predictions", "precision", "f1_score", "accuracy"]
    )


def test_unload_model_error(app):
    project_err = "/models?project=my_project&model=my_model"
    _, response = app.delete(project_err)
    rjs = response.json
    assert response.status == 500, "Project not found"
    assert rjs["details"]["error"] == "Project my_project could not be found"


def test_unload_model(app):
    unload = "/models?model={}&project={}".format(
        KEYWORD_MODEL_NAME, KEYWORD_PROJECT_NAME
    )
    _, response = app.delete(unload)
    rjs = response.json
    assert response.status == 200, "Fallback model unloaded"
    assert rjs == KEYWORD_MODEL_NAME
