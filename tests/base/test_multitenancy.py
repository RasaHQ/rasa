# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import pytest
from treq.testing import StubTreq

from rasa_nlu import config
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.data_router import DataRouter
from rasa_nlu.model import Trainer
from rasa_nlu.server import RasaNLU
from tests.utilities import ResponseTest


@pytest.fixture(scope="module")
def app(component_builder):
    """Use IResource interface of Klein to mock Rasa HTTP server.

    :param component_builder:
    :return:
    """

    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")

    train_models(component_builder,
                 os.path.join(root_dir, "data/examples/rasa/demo-rasa.json"))

    router = DataRouter(os.path.join(root_dir, "test_projects"))
    rasa = RasaNLU(router, logfile=nlu_log_file, testing=True)

    return StubTreq(rasa.app.resource())


@pytest.mark.parametrize("response_test", [
    ResponseTest(
            "http://dummy-uri/parse?q=food&project=test_project_mitie",
            {"entities": [], "intent": "affirm", "text": "food"}
    ),
    ResponseTest(
            "http://dummy-uri/parse?q=food&project=test_project_mitie_sklearn",
            {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
    ResponseTest(
            "http://dummy-uri/parse?q=food&project=test_project_spacy_sklearn",
            {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
])
@pytest.inlineCallbacks
def test_get_parse(app, response_test):
    response = yield app.get(response_test.endpoint)
    rjs = yield response.json()

    assert response.code == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
            "http://dummy-uri/parse?q=food",
            {"error": "No project found with name 'default'."}
    ),
    ResponseTest(
            "http://dummy-uri/parse?q=food&project=umpalumpa",
            {"error": "No project found with name 'umpalumpa'."}
    )
])
@pytest.inlineCallbacks
def test_get_parse_invalid_model(app, response_test):
    response = yield app.get(response_test.endpoint)
    rjs = yield response.json()
    assert response.code == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
            "http://dummy-uri/parse",
            {"entities": [], "intent": "affirm", "text": "food"},
            payload={"q": "food", "project": "test_project_mitie"}
    ),
    ResponseTest(
            "http://dummy-uri/parse",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
            payload={"q": "food", "project": "test_project_mitie_sklearn"}
    ),
    ResponseTest(
            "http://dummy-uri/parse",
            {"entities": [], "intent": "restaurant_search", "text": "food"},
            payload={"q": "food", "project": "test_project_spacy_sklearn"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint, json=response_test.payload)
    rjs = yield response.json()
    assert response.code == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


@pytest.inlineCallbacks
def test_post_parse_specific_model(app):
    status = yield app.get("http://dummy-uri/status")
    sjs = yield status.json()
    project = sjs["available_projects"]["test_project_spacy_sklearn"]
    model = project["available_models"][0]

    query = ResponseTest("http://dummy-uri/parse",
                         {"entities": [], "intent": "affirm", "text": "food"},
                         payload={"q": "food",
                                  "project": "test_project_spacy_sklearn",
                                  "model": model})

    response = yield app.post(query.endpoint, json=query.payload)
    assert response.code == 200

    # check that that model now is loaded in the server
    status = yield app.get("http://dummy-uri/status")
    sjs = yield status.json()
    project = sjs["available_projects"]["test_project_spacy_sklearn"]
    assert model in project["loaded_models"]


@pytest.mark.parametrize("response_test", [
    ResponseTest(
            "http://dummy-uri/parse",
            {"error": "No project found with name 'default'."},
            payload={"q": "food"}
    ),
    ResponseTest(
            "http://dummy-uri/parse",
            {"error": "No project found with name 'umpalumpa'."},
            payload={"q": "food", "project": "umpalumpa"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse_invalid_model(app, response_test):
    response = yield app.post(response_test.endpoint, json=response_test.payload)
    rjs = yield response.json()
    assert response.code == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


def train_models(component_builder, data):
    # Retrain different multitenancy models
    def train(cfg_name, project_name):
        from rasa_nlu.train import create_persistor
        from rasa_nlu import training_data

        cfg = config.load(cfg_name)
        trainer = Trainer(cfg, component_builder)
        training_data = training_data.load_data(data)

        trainer.train(training_data)
        trainer.persist("test_projects", project_name=project_name)

    train("sample_configs/config_spacy.yml", "test_project_spacy_sklearn")
    train("sample_configs/config_mitie.yml", "test_project_mitie")
    train("sample_configs/config_mitie_sklearn.yml", "test_project_mitie_sklearn")
