# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import json
import os
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.server import create_app
from utilities import ResponseTest


@pytest.fixture(scope="module")
def app(component_builder):
    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "backend": "mitie",
        "path": os.path.join(root_dir, "test_models"),
        "data": os.path.join(root_dir, "data/demo-restaurants.json")
    }
    train_models(component_builder)
    config = RasaNLUConfig(cmdline_args=_config)
    application = create_app(config, component_builder)
    return application


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food&agent=test_model_mitie",
        {"entities": [], "intent": "affirm", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food&agent=test_model_mitie_sklearn",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
    ResponseTest(
        "/parse?q=food&agent=test_model_spacy_sklearn",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
])
def test_get_parse(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 200
    assert all(prop in response.json for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse?q=food",
        {"error": "No agent found with name 'default'."}
    ),
    ResponseTest(
        "/parse?q=food&agent=umpalumpa",
        {"error": "No agent found with name 'umpalumpa'."}
    )
])
def test_get_parse_invalid_model(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 404
    assert response.json.get("error").startswith(response_test.expected_response["error"])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "affirm", "text": "food"},
        payload={"q": "food", "agent": "test_model_mitie"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "agent": "test_model_mitie_sklearn"}
    ),
    ResponseTest(
        "/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "agent": "test_model_spacy_sklearn"}
    ),
])
def test_post_parse(client, response_test):
    response = client.post(response_test.endpoint, data=json.dumps(response_test.payload),
                           content_type='application/json')
    assert response.status_code == 200
    assert all(prop in response.json for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        {"error": "No agent found with name 'default'."},
        payload={"q": "food"}
    ),
    ResponseTest(
        "/parse",
        {"error": "No agent found with name 'umpalumpa'."},
        payload={"q": "food", "agent": "umpalumpa"}
    ),
])
def test_post_parse_invalid_model(client, response_test):
    response = client.post(response_test.endpoint, data=json.dumps(response_test.payload),
                           content_type='application/json')
    assert response.status_code == 404
    assert response.json.get("error").startswith(response_test.expected_response["error"])


def train_models(component_builder):
    # Retrain different multitenancy models
    def train(cfg_name, model_name):
        from rasa_nlu.train import create_persistor
        from rasa_nlu.converters import load_data

        config = RasaNLUConfig(cfg_name)
        trainer = Trainer(config, component_builder)
        training_data = load_data(config['data'])

        trainer.train(training_data)
        persistor = create_persistor(config)
        trainer.persist("test_models", persistor, agent_name=model_name)

    train("config_mitie.json", "test_model_mitie")
    train("config_spacy.json", "test_model_spacy_sklearn")
    train("config_mitie_sklearn.json", "test_model_mitie_sklearn")
