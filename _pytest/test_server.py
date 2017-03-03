# -*- coding: utf-8 -*-
import tempfile

import pytest

from rasa_nlu.config import RasaNLUConfig
import json
import codecs

from helpers import ResponseTest
from rasa_nlu.server import setup_app


@pytest.fixture
def app():
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': -1,  # unused in test app
        "backend": "mitie",
        "path": "./",
        "data": "./data/demo-restaurants.json",
        "emulate": "wit"
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = setup_app(config)
    return application


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200 and response.data == "hello"


def test_status(client):
    response = client.get("/status")
    rjs = response.json
    assert response.status_code == 200 and ("trainings_under_this_process" in rjs and "available_models" in rjs)


def test_get_parse(client):
    tests = [
        ResponseTest(
            u"/parse?q=hello",
            [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello"}]
        ),
        ResponseTest(
            u"/parse?q=hello ńöñàśçií",
            [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello ńöñàśçií"}]
        ),
    ]
    for test in tests:
        response = client.get(test.endpoint)
        assert response.status_code == 200 and response.json == test.expected_response


def test_post_parse(client):
    tests = [
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
    ]
    for test in tests:
        response = client.post(test.endpoint, data=json.dumps(test.payload), content_type='application/json')
        assert response.status_code == 200 and response.json == test.expected_response


def test_post_train(client):
    with codecs.open('data/examples/luis/demo-restaurants.json',
                     encoding='utf-8') as train_file:
        train_data = json.loads(train_file.read())
    response = client.post("/train", data=json.dumps(train_data), content_type='application/json')
    assert response.status_code == 200
