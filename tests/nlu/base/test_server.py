# -*- coding: utf-8 -*-
import asyncio
import io
import json
import os
import shutil
import tarfile
import tempfile
import time

import pytest
import ruamel.yaml as yaml
import tempfile

from rasa import data, model
from rasa.cli.utils import create_output_path
from rasa.model import unpack_model
from rasa.nlu import config
from rasa.nlu.data_router import DataRouter
from rasa.nlu.model import Trainer
from rasa.nlu.server import create_app
from tests.nlu import utilities
from tests.nlu.utilities import ResponseTest


@pytest.fixture(scope="module")
def router(component_builder):
    """Test sanic server."""

    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    train_models(
        component_builder, os.path.join(root_dir, "data/examples/rasa/demo-rasa.json")
    )

    return DataRouter(os.path.join(root_dir, "test_projects/test_project_spacy"))


@pytest.fixture(scope="module")
def router_without_model(component_builder):
    """Test sanic server."""
    return DataRouter("not-existing-dir")


@pytest.fixture
def app(router):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    rasa = create_app(router, logfile=nlu_log_file)

    return rasa.test_client


@pytest.fixture
def app_without_model(router_without_model):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    rasa = create_app(router_without_model, logfile=nlu_log_file)

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
    assert response.status == 200 and content.startswith("hello")


def test_status(app):
    _, response = app.get("/status")
    rjs = response.json
    assert response.status == 200
    assert "loaded_model" in rjs
    assert "current_training_processes" in rjs
    assert "max_training_processes" in rjs


def test_version(app):
    _, response = app.get("/version")
    rjs = response.json
    assert response.status == 200
    assert set(rjs.keys()) == {"version", "minimum_compatible_version"}


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=hello",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?query=hello",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello ńöñàśçií",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
        ),
        ResponseTest(
            "/parse?q=",
            {"entities": [], "intent": {"confidence": 0.0, "name": None}, "text": ""},
        ),
    ],
)
def test_get_parse(app, response_test):
    _, response = app.get(response_test.endpoint)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["model"].startswith("nlu")
    assert rjs["text"] == response_test.expected_response["text"]


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"q": "hello"},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"query": "hello"},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"q": "hello ńöñàśçií"},
        ),
    ],
)
def test_post_parse(app, response_test):
    _, response = app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["model"].startswith("nlu")
    assert rjs["text"] == response_test.expected_response["text"]


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=hello&model=some-model",
            {
                "entities": [],
                "model": "fallback",
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?query=hello&model=some-model",
            {
                "entities": [],
                "model": "fallback",
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello ńöñàśçií&model=some-model",
            {
                "entities": [],
                "model": "fallback",
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
        ),
        ResponseTest(
            "/parse?q=&model=abc",
            {
                "entities": [],
                "model": "fallback",
                "intent": {"confidence": 0.0, "name": None},
                "text": "",
            },
        ),
    ],
)
def test_get_parse_use_fallback_model(app_without_model, response_test):
    _, response = app_without_model.get(response_test.endpoint)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["model"] == "fallback"
    assert rjs["text"] == response_test.expected_response["text"]


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"q": "hello", "model": "some-model"},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"query": "hello", "model": "some-model"},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"q": "hello ńöñàśçií", "model": "some-model"},
        ),
    ],
)
def test_post_parse_using_fallback_model(app, response_test):
    _, response = app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text", "model"])
    assert rjs["entities"] == response_test.expected_response["entities"]
    assert rjs["model"] == "fallback"
    assert rjs["text"] == response_test.expected_response["text"]
    assert rjs["intent"]["name"] == response_test.expected_response["intent"]["name"]


@utilities.slowtest
def test_post_train_success(app_without_model, rasa_default_train_data):
    request = {
        "language": "en",
        "pipeline": "pretrained_embeddings_spacy",
        "data": rasa_default_train_data,
    }

    _, response = app_without_model.post("/train", json=request)

    assert response.status == 200
    assert response.content is not None


@utilities.slowtest
def test_post_train_internal_error(app, rasa_default_train_data):
    _, response = app.post(
        "/train", json={"data": "dummy_data_for_triggering_an_error"}
    )
    rjs = response.json
    assert response.status == 500, "The training data format is not valid"
    assert "error" in rjs


def test_model_hot_reloading(app, rasa_default_train_data):
    query = "/parse?q=hello&model=test-model"
    _, response = app.get(query)
    assert response.status == 200
    rjs = response.json
    assert rjs["model"] == "fallback"

    train_u = "/train?model=test-model"
    request = {
        "language": "en",
        "pipeline": "pretrained_embeddings_spacy",
        "data": rasa_default_train_data,
    }
    model_str = yaml.safe_dump(request, default_flow_style=False, allow_unicode=True)
    _, response = app.post(
        train_u, headers={"Content-Type": "application/x-yml"}, data=model_str
    )
    assert response.status == 200, "Training should end successfully"

    _, response = app.post(
        train_u, headers={"Content-Type": "application/json"}, data=json.dumps(request)
    )
    assert response.status == 200, "Training should end successfully"

    _, response = app.get("/parse?q=hello&model=test-model")
    assert response.status == 200
    rjs = response.json
    assert "test-model" in rjs["model"]


def test_evaluate_invalid_project_error(app, rasa_default_train_data):
    _, response = app.post("/evaluate?model=not-existing", json=rasa_default_train_data)

    rjs = response.json
    assert response.status == 500
    assert "error" in rjs
    assert rjs["error"] == "Model with name 'not-existing' is not loaded."


def test_evaluate_internal_error(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate", json={"data": "dummy_data_for_triggering_an_error"}
    )

    rjs = response.json
    assert response.status == 500, "The training data format is not valid"
    assert "error" in rjs
    assert "Unknown data format for file" in rjs["error"]


def test_evaluate(app, rasa_default_train_data):
    _, response = app.post("/evaluate", json=rasa_default_train_data)

    rjs = response.json
    assert response.status == 200, "Evaluation should start"
    assert "intent_evaluation" in rjs
    assert "entity_evaluation" in rjs
    assert all(
        prop in rjs["intent_evaluation"]
        for prop in ["report", "predictions", "precision", "f1_score", "accuracy"]
    )


def test_unload_model_error(app):
    project_err = "/models?model=my_model"
    _, response = app.delete(project_err)
    rjs = response.json
    assert (
        response.status == 500
    ), "Model is not loaded and can therefore not be unloaded."
    assert rjs["error"] == "Model with name 'my_model' is not loaded."


def test_unload_model(app):
    unload = "/models"
    _, response = app.delete(unload)
    assert response.status == 204, "No Content"


def train_models(component_builder, data_path):
    # Retrain different multitenancy models
    def train(cfg_name, sub_folder):
        from rasa.nlu import training_data

        cfg = config.load(cfg_name)
        trainer = Trainer(cfg, component_builder)
        training_data = training_data.load_data(data_path)

        trainer.train(training_data)

        model_dir = os.path.join("test_projects", sub_folder)
        model_path = trainer.persist(model_dir)

        nlu_data = data.get_nlu_directory(data_path)
        output_path = create_output_path(model_dir, prefix="nlu-")
        new_fingerprint = model.model_fingerprint(cfg_name, nlu_data=nlu_data)
        model.create_package_rasa(model_path, output_path, new_fingerprint)

    if os.path.exists("test_projects"):
        shutil.rmtree("test_projects")

    train("sample_configs/config_pretrained_embeddings_spacy.yml", "test_project_spacy")
