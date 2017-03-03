# -*- coding: utf-8 -*-
import json
import os
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.server import setup_app
from utilities import ResponseTest


@pytest.fixture(scope="module")
def app():
    if "TRAVIS_BUILD_DIR" in os.environ:
        root_dir = os.environ["TRAVIS_BUILD_DIR"]
    else:
        root_dir = os.getcwd()
        print("model at")
        print(os.path.join(root_dir, "models/model_1"))

    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "backend": "mitie",
        "path": root_dir,
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "server_model_dir": {
            "one": os.path.join(root_dir, "test_models/test_model_mitie"),
            "two": os.path.join(root_dir, "test_models/test_model_mitie_sklearn"),
            "three": os.path.join(root_dir, "test_models/test_model_spacy_sklearn")
        }
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = setup_app(config)
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
        {u"error": u"no model found with alias: default"}
    ),
    ResponseTest(
        u"/parse?q=food&model=umpalumpa",
        {u"error": u"no model found with alias: umpalumpa"}
    )
])
def test_get_parse_invalid_model(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 200
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
        {u"error": u"no model found with alias: default"},
        payload={u"q": u"food"}
    ),
    ResponseTest(
        u"/parse",
        {u"error": u"no model found with alias: umpalumpa"},
        payload={u"q": u"food", u"model": u"umpalumpa"}
    ),
])
def test_post_parse_invalid_model(client, response_test):
    response = client.post(response_test.endpoint,
                           data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 200
    assert response.json == response_test.expected_response
