# -*- coding: utf-8 -*-
import asyncio
import io
import json

import pytest
import ruamel.yaml as yaml
import tempfile

from rasa.nlu.data_router import DataRouter, create_data_router
from rasa.nlu.model_loader import FALLBACK_MODEL_NAME
from rasa.nlu.server import create_app
from tests.nlu import utilities
from tests.nlu.conftest import NLU_MODEL_PATH, NLU_MODEL_NAME
from tests.nlu.utilities import ResponseTest


@pytest.fixture
def app_without_model():
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    router = DataRouter()
    rasa = create_app(router, logfile=nlu_log_file)

    return rasa.test_client


@pytest.fixture()
async def app(tmpdir_factory, trained_nlu_model):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    data_router = await create_data_router(NLU_MODEL_PATH)
    rasa = create_app(data_router, logfile=nlu_log_file)

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
    assert response.status == 200
    assert "loaded_model" in rjs
    assert "current_worker_processes" in rjs
    assert "max_worker_processes" in rjs


def test_version(app):
    _, response = app.get("/version")
    rjs = response.json
    assert response.status == 200
    assert set(rjs.keys()) == {"version", "minimum_compatible_version"}


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=hello&model={}".format(NLU_MODEL_NAME),
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello&model={}".format(NLU_MODEL_NAME),
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
        ),
        ResponseTest(
            "/parse?q=hello ńöñàśçií&model={}".format(NLU_MODEL_NAME),
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
        ),
        ResponseTest(
            "/parse?q=&model={}".format(NLU_MODEL_NAME),
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
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
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"q": "hello", "model": NLU_MODEL_NAME},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello",
            },
            payload={"query": "hello", "model": NLU_MODEL_NAME},
        ),
        ResponseTest(
            "/parse",
            {
                "entities": [],
                "model": NLU_MODEL_NAME,
                "intent": {"confidence": 1.0, "name": "greet"},
                "text": "hello ńöñàśçií",
            },
            payload={"q": "hello ńöñàśçií", "model": NLU_MODEL_NAME},
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
    assert rjs["model"] == FALLBACK_MODEL_NAME
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
    assert rjs["model"] == FALLBACK_MODEL_NAME
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
    assert response.status == 500, "The training data format is not valid"


def test_model_hot_reloading(app, rasa_default_train_data):
    query = "/parse?q=hello&model=test-model"

    # Model could not be found, fallback model was used instead
    _, response = app.get(query)
    assert response.status == 200
    rjs = response.json
    assert rjs["model"] == FALLBACK_MODEL_NAME

    # Train a new model - model will be loaded automatically
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

    # Model should be there now
    _, response = app.get(query)
    assert response.status == 200, "Model should now exist after it got trained"
    rjs = response.json
    assert "test-model" in rjs["model"]


def test_evaluate_invalid_model_error(app, rasa_default_train_data):
    _, response = app.post("/evaluate?model=not-existing", json=rasa_default_train_data)

    rjs = response.json
    assert response.status == 500
    assert "details" in rjs
    assert rjs["details"]["error"] == "Model with name 'not-existing' is not loaded."


def test_evaluate_unsupported_model_error(app_without_model, rasa_default_train_data):
    _, response = app_without_model.post("/evaluate", json=rasa_default_train_data)

    rjs = response.json
    assert response.status == 500
    assert "details" in rjs
    assert rjs["details"]["error"] == "No model is loaded. Cannot evaluate."


def test_evaluate_internal_error(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate", json={"data": "dummy_data_for_triggering_an_error"}
    )
    assert response.status == 500, "The training data format is not valid"


def test_evaluate(app, rasa_default_train_data):
    _, response = app.post(
        "/evaluate?model={}".format(NLU_MODEL_NAME), json=rasa_default_train_data
    )

    rjs = response.json
    assert "intent_evaluation" in rjs
    assert "entity_evaluation" in rjs
    assert all(
        prop in rjs["intent_evaluation"]
        for prop in ["report", "predictions", "precision", "f1_score", "accuracy"]
    )
    assert response.status == 200, "Evaluation should start"


def test_unload_model_error(app):
    request = "/models?model=my_model"
    _, response = app.delete(request)
    rjs = response.json
    assert (
        response.status == 404
    ), "Model is not loaded and can therefore not be unloaded."
    assert rjs["details"]["error"] == "Model with name 'my_model' is not loaded."


def test_unload_model(app):
    unload = "/models?model={}".format(NLU_MODEL_NAME)
    _, response = app.delete(unload)
    assert response.status == 204, "No Content"


def test_status_after_unloading(app):
    _, response = app.get("/status")
    rjs = response.json
    assert response.status == 200
    assert rjs["loaded_model"] == NLU_MODEL_NAME

    unload = "/models?model={}".format(NLU_MODEL_NAME)
    _, response = app.delete(unload)
    assert response.status == 204, "No Content"

    _, response = app.get("/status")
    rjs = response.json
    assert response.status == 200
    assert rjs["loaded_model"] is None
