# -*- coding: utf-8 -*-
import os
import shutil

import pytest
import tempfile

from rasa import model, data
from rasa.cli.utils import create_output_path
from rasa.nlu import config
from rasa.nlu.data_router import DataRouter
from rasa.nlu.model import Trainer
from rasa.nlu.server import create_app
from tests.nlu.utilities import ResponseTest


@pytest.fixture(scope="module")
def router(component_builder):
    """Test sanic server."""

    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    train_models(component_builder,
                 os.path.join(root_dir, "data/examples/rasa/demo-rasa.json"))

    router = DataRouter(os.path.join(root_dir, "test_projects/test_project_mitie"))
    return router


@pytest.fixture
def app(router):
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    rasa = create_app(
        router,
        logfile=nlu_log_file)

    return rasa.test_client


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food",
        {"entities": [], "intent": "affirm", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
])
def test_get_parse(app, response_test):
    _, response = app.get(response_test.endpoint)
    rjs = response.json

    assert response.status == 200
    assert all(prop in rjs
               for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food&model=default",
        {"error": "No model loaded with name 'default'."}
    )
])
def test_get_parse_invalid_model(app, response_test):
    _, response = app.get(response_test.endpoint)
    rjs = response.json
    assert response.status == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "affirm", "text": "food"},
        payload={"q": "food", "project": "test_project_mitie"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "project": "test_project_mitie_2"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "project": "test_project_spacy"}
    ),
])
def test_post_parse(app, response_test):
    _, response = app.post(response_test.endpoint,
                           json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


def test_post_parse_specific_model(app):
    _, status = app.get("/status")
    sjs = status.json
    model = sjs["loaded_model"]

    query = ResponseTest("/parse",
                         {"entities": [], "intent": "affirm", "text": "food"},
                         payload={"q": "food",
                                  "model": model})

    _, response = app.post(query.endpoint, json=query.payload)
    assert response.status == 200

    # check that that model now is loaded in the server
    _, status = app.get("/status")
    sjs = status.json
    assert model == sjs["loaded_model"]


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"error": "No model loaded with name 'default'."},
        payload={"q": "food", "model": "default"}
    )
])
def test_post_parse_invalid_model(app, response_test):
    _, response = app.post(response_test.endpoint,
                           json=response_test.payload)
    rjs = response.json
    assert response.status == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


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

    train("sample_configs/config_pretrained_embeddings_spacy.yml",
          "test_project_spacy")
    train("sample_configs/config_pretrained_embeddings_mitie.yml",
          "test_project_mitie")
    train("sample_configs/config_pretrained_embeddings_mitie_2.yml",
          "test_project_mitie_2")
