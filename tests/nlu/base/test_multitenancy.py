# -*- coding: utf-8 -*-
import os
import pytest
import tempfile


from rasa.nlu import config
from rasa.nlu.data_router import DataRouter
from rasa.nlu.model import Trainer
from rasa.nlu.server import create_app
from tests.nlu.utilities import ResponseTest


@pytest.fixture(scope="module")
def router(component_builder):
    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    train_models(
        component_builder, os.path.join(root_dir, "data/examples/rasa/demo-rasa.json")
    )

    router = DataRouter(os.path.join(root_dir, "test_projects"))
    return router


@pytest.fixture
def app(router):
    """Test client for the http server."""

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    rasa = create_app(router, logfile=nlu_log_file)

    return rasa.test_client


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=food&project=test_project_mitie",
            {"entities": [], "intent": "affirm", "text": "food"},
        ),
        ResponseTest(
            "/parse?q=food&project=test_project_mitie_2",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
        ),
        ResponseTest(
            "/parse?q=food&project=test_project_spacy",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
        ),
    ],
)
def test_get_parse(app, response_test):
    _, response = app.get(response_test.endpoint)
    rjs = response.json

    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse?q=food", {"error": "No project found with name 'default'."}
        ),
        ResponseTest(
            "/parse?q=food&project=umpalumpa",
            {"error": "No project found with name 'umpalumpa'."},
        ),
    ],
)
def test_get_parse_invalid_model(app, response_test):
    _, response = app.get(response_test.endpoint)

    assert response.status == 404


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse",
            {"entities": [], "intent": "affirm", "text": "food"},
            payload={"q": "food", "project": "test_project_mitie"},
        ),
        ResponseTest(
            "/parse",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
            payload={"q": "food", "project": "test_project_mitie_2"},
        ),
        ResponseTest(
            "/parse",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
            payload={"q": "food", "project": "test_project_spacy"},
        ),
    ],
)
def test_post_parse(app, response_test):
    _, response = app.post(response_test.endpoint, json=response_test.payload)
    rjs = response.json
    assert response.status == 200
    assert all(prop in rjs for prop in ["entities", "intent", "text"])


def test_post_parse_specific_model(app):
    _, status = app.get("/status")
    sjs = status.json
    project = sjs["available_projects"]["test_project_spacy"]
    model = project["available_models"][-1]

    query = ResponseTest(
        "/parse",
        {"entities": [], "intent": "affirm", "text": "food"},
        payload={"q": "food", "project": "test_project_spacy", "model": model},
    )

    _, response = app.post(query.endpoint, json=query.payload)
    assert response.status == 200

    # check that that model now is loaded in the server
    _, status = app.get("/status")
    sjs = status.json
    project = sjs["available_projects"]["test_project_spacy"]
    assert model in project["loaded_models"]


@pytest.mark.parametrize(
    "response_test",
    [
        ResponseTest(
            "/parse",
            {"error": "No project found with name 'default'."},
            payload={"q": "food"},
        ),
        ResponseTest(
            "/parse",
            {"error": "No project found with name 'umpalumpa'."},
            payload={"q": "food", "project": "umpalumpa"},
        ),
    ],
)
def test_post_parse_invalid_project(app, response_test):
    _, response = app.post(response_test.endpoint, json=response_test.payload)
    assert response.status == 404


def train_models(component_builder, data):
    # Retrain different multitenancy models
    def train(cfg_name, project_name):
        from rasa.nlu import training_data

        cfg = config.load(cfg_name)
        trainer = Trainer(cfg, component_builder)
        training_data = training_data.load_data(data)

        trainer.train(training_data)
        trainer.persist("test_projects", project_name=project_name)

    train("sample_configs/config_pretrained_embeddings_spacy.yml", "test_project_spacy")
    train("sample_configs/config_pretrained_embeddings_mitie.yml", "test_project_mitie")
    train(
        "sample_configs/config_pretrained_embeddings_mitie_2.yml",
        "test_project_mitie_2",
    )
