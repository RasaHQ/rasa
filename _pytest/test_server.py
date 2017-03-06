# -*- coding: utf-8 -*-
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
import json
import codecs

from utilities import ResponseTest
from rasa_nlu.server import create_app


@pytest.fixture(scope="module")
def app():
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,                 # unused in test app
        "backend": "mitie",
        "path": "./models",
        "server_model_dirs": {},
        "data": "./data/demo-restaurants.json",
        "emulate": "wit",
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = create_app(config)
    return application


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200 and response.data == "hello"


def test_status(client):
    response = client.get("/status")
    rjs = response.json
    assert response.status_code == 200 and \
        ("trainings_under_this_process" in rjs and "available_models" in rjs)


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        u"/parse?q=hello",
        [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello"}]
    ),
    ResponseTest(
        u"/parse?q=hello ńöñàśçií",
        [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello ńöñàśçií"}]
    ),
])
def test_get_parse(client, response_test):
    response = client.get(response_test.endpoint)
    assert response.status_code == 200
    assert len(response.json) == 1
    assert all(prop in response.json[0] for prop in ['entities', 'intent', '_text', 'confidence'])


@pytest.mark.parametrize("response_test", [
    ResponseTest(
        "/parse",
        [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello"}],
        payload={u"q": u"hello"}
    ),
    ResponseTest(
        "/parse",
        [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello ńöñàśçií"}],
        payload={u"q": u"hello ńöñàśçií"}
    ),
])
def test_post_parse(client, response_test):
    response = client.post(response_test.endpoint,
                           data=json.dumps(response_test.payload), content_type='application/json')
    assert response.status_code == 200
    assert len(response.json) == 1
    assert all(prop in response.json[0] for prop in ['entities', 'intent', '_text', 'confidence'])


def test_post_train(client):
    with codecs.open('data/examples/luis/demo-restaurants.json',
                     encoding='utf-8') as train_file:
        train_data = json.loads(train_file.read())
    response = client.post("/train", data=json.dumps(train_data), content_type='application/json')
    assert response.status_code == 200
