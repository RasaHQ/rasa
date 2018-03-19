# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import tempfile

import pytest
from treq.testing import StubTreq

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.server import RasaNLU
from tests.utilities import ResponseTest


@pytest.fixture(scope="module")
def app(component_builder):
    """
    This fixture makes use of the IResource interface of the Klein application to mock Rasa HTTP server.
    :param component_builder:
    :return:
    """

    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "pipeline": "keyword",

        "path": os.path.join(root_dir, "test_projects"),
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "max_training_processes": 1
    }
    train_models(component_builder)

    config = RasaNLUConfig(cmdline_args=_config)
    rasa = RasaNLU(config, component_builder, True)
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
    model = sjs["available_projects"]["test_project_mitie"]["available_models"][0]
    query = ResponseTest("http://dummy-uri/parse", {"entities": [], "intent": "affirm", "text": "food"},
                         payload={"q": "food", "project": "test_project_mitie", "model": model})
    response = yield app.post(query.endpoint, json=query.payload)
    assert response.code == 200


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


def train_models(component_builder):
    # Retrain different multitenancy models
    def train(cfg_name, project_name):
        from rasa_nlu.train import create_persistor
        from rasa_nlu import training_data

        config = RasaNLUConfig(cfg_name)
        trainer = Trainer(config, component_builder)
        training_data = training_data.load_data(config['data'])

        trainer.train(training_data)
        persistor = create_persistor(config)
        trainer.persist("test_projects", persistor, project_name)

    train("sample_configs/config_mitie.json", "test_project_mitie")
    train("sample_configs/config_spacy.json", "test_project_spacy_sklearn")
    train("sample_configs/config_mitie_sklearn.json", "test_project_mitie_sklearn")
