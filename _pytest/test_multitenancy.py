# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json

import os
import tempfile

import pytest

from treq.testing import StubTreq

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.model import Trainer
from rasa_nlu.server import RasaNLU
from utilities import ResponseTest


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
        "path": os.path.join(root_dir, "test_models"),
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "server_model_dirs": {
            "one": "test_model_mitie",
            "two": "test_model_mitie_sklearn",
            "three": "test_model_spacy_sklearn",
        },
        "max_training_processes": 1
    }

    config = RasaNLUConfig(cmdline_args=_config)
    rasa = RasaNLU(config, component_builder, True)
    return StubTreq(rasa.app.resource())


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy_uri/parse?q=food&model=one",
        {"entities": [], "intent": "affirm", "text": "food"}
    ),
    ResponseTest(
        "http://dummy_uri/parse?q=food&model=two",
        {"entities": [], "intent": "restaurant_search", "text": "food"}
    ),
    ResponseTest(
        "http://dummy_uri/parse?q=food&model=three",
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
        "http://dummy_uri/parse?q=food",
        {"error": "No model found with alias 'default'."}
    ),
    ResponseTest(
        "http://dummy_uri/parse?q=food&model=umpalumpa",
        {"error": "No model found with alias 'umpalumpa'."}
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
        "http://dummy_uri/parse",
        {"entities": [], "intent": "affirm", "text": "food"},
        payload={"q": "food", "model": "one"}
    ),
    ResponseTest(
        "http://dummy_uri/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "model": "two"}
    ),
    ResponseTest(
        "http://dummy_uri/parse",
        {"entities": [], "intent": "restaurant_search", "text": "food"},
        payload={"q": "food", "model": "three"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse(app, response_test):
    response = yield app.post(response_test.endpoint, data=json.dumps(response_test.payload),
                              content_type='application/json')
    rjs = yield response.json()
    assert response.code == 200
    assert all(prop in rjs for prop in ['entities', 'intent', 'text'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "http://dummy_uri/parse",
        {"error": "No model found with alias 'default'."},
        payload={"q": "food"}
    ),
    ResponseTest(
        "http://dummy_uri/parse",
        {"error": "No model found with alias 'umpalumpa'."},
        payload={"q": "food", "model": "umpalumpa"}
    ),
])
@pytest.inlineCallbacks
def test_post_parse_invalid_model(app, response_test):
    response = yield app.post(response_test.endpoint, data=json.dumps(response_test.payload),
                              content_type='application/json')
    rjs = yield response.json()
    assert response.code == 404
    assert rjs.get("error").startswith(response_test.expected_response["error"])


if __name__ == '__main__':
    # Retrain different multitenancy models
    def train(cfg_name, model_name):
        from rasa_nlu.train import create_persistor
        from rasa_nlu.converters import load_data

        config = RasaNLUConfig(cfg_name)
        trainer = Trainer(config)
        training_data = load_data(config['data'])

        trainer.train(training_data)
        persistor = create_persistor(config)
        trainer.persist("test_models", persistor, model_name=model_name)

    train("sample_configs/config_mitie.json", "test_model_mitie")
    train("sample_configs/config_spacy.json", "test_model_spacy_sklearn")
    train("sample_configs/config_mitie_sklearn.json", "test_model_mitie_sklearn")
