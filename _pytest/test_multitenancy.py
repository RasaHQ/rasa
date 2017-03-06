# -*- coding: utf-8 -*-
import json
import os
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.server import create_app
from utilities import ResponseTest


@pytest.fixture(scope="session")
def app():
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
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "server_model_dirs": {
            "one": "test_model_mitie",
            "two": "test_model_mitie_sklearn",
            "three": "test_model_spacy_sklearn",
        }
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = create_app(config)
    return application


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        u"/parse?q=food&model=one",
        {u"entities": [], u"intent": u"affirm", u"text": u"food"}
    ),
    ResponseTest(
        u"/parse?q=food&model=two",
        {u"entities": [], u"intent": u"restaurant_search", u"text": u"food"}
    ),
    ResponseTest(
        u"/parse?q=food&model=three",
        {u"entities": [], u"intent": u"restaurant_search", u"text": u"food"}
    ),
])
def test_get_parse(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 200
    assert all(prop in response.json for prop in ['entities', 'intent', 'text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        u"/parse?q=food",
        {u"error": u"No model found with alias 'default'"}
    ),
    ResponseTest(
        u"/parse?q=food&model=umpalumpa",
        {u"error": u"No model found with alias 'umpalumpa'"}
    )
])
def test_get_parse_invalid_model(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 404
    assert response.json == response_test.expected_response


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        u"/parse",
        {u"entities": [], u"intent": u"affirm", u"text": u"food"},
        payload={u"q": u"food", u"model": "one"}
    ),
    ResponseTest(
        u"/parse",
        {u"entities": [], u"intent": u"restaurant_search", u"text": u"food"},
        payload={u"q": u"food", u"model": "two"}
    ),
    ResponseTest(
        u"/parse",
        {u"entities": [], u"intent": u"restaurant_search", u"text": u"food"},
        payload={u"q": u"food", u"model": "three"}
    ),
])
def test_post_parse(client, response_test):
    response = client.post(response_test.endpoint,
                           data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 200
    assert all(prop in response.json for prop in ['entities', 'intent', 'text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        u"/parse",
        {u"error": u"No model found with alias 'default'"},
        payload={u"q": u"food"}
    ),
    ResponseTest(
        u"/parse",
        {u"error": u"No model found with alias 'umpalumpa'"},
        payload={u"q": u"food", u"model": u"umpalumpa"}
    ),
])
def test_post_parse_invalid_model(client, response_test):
    response = client.post(response_test.endpoint,
                           data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 404
    assert response.json == response_test.expected_response


if __name__ == '__main__':
    # Retrain different multitenancy models
    def train(cfg_name, model_name):
        from rasa_nlu.train import create_trainer
        from rasa_nlu.train import create_persistor
        from rasa_nlu.training_data import TrainingData

        config = RasaNLUConfig(cfg_name)
        trainer = create_trainer(config)
        persistor = create_persistor(config)

        training_data = TrainingData(config['data'], config['backend'], nlp=trainer.nlp)
        trainer.train(training_data)
        trainer.persist(os.path.join("test_models", model_name), persistor, create_unique_subfolder=False)

    train("config_mitie.json", "test_model_mitie")
    train("config_spacy.json", "test_model_spacy_sklearn")
    train("config_mitie_sklearn.json", "test_model_mitie_sklearn")
