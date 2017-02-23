# -*- coding: utf-8 -*-
import tempfile

import pytest
import requests
import os

from helpers import ResponseTest
from rasa_nlu.server import setup_app
from rasa_nlu.config import RasaNLUConfig
from multiprocessing import Process
import time
import json
import codecs


@pytest.fixture
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
            "one": os.path.join(root_dir, "models/model_1"),
            "two": os.path.join(root_dir, "models/model_2")
        }
    }
    config = RasaNLUConfig(cmdline_args=_config)
    application = setup_app(config)
    return application


def test_get_parse(client):
    tests = [
        ResponseTest(
            u"/parse?q=food&model=one",
            {u"entities": [], u"intent": u"affirm", u"text": u"food"}
        ),
        ResponseTest(
            u"/parse?q=food&model=two",
            {u"entities": [], u"intent": u"restaurant_search", u"text": u"food"}
        ),
        ResponseTest(
            u"/parse?q=food",
            {u"error": u"no model found with alias: default"}
        ),
    ]
    for test in tests:
        response = client.get(test.endpoint)
        assert response.status_code == 200 and response.json == test.expected_response


def test_post_parse(client):
    tests = [
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
            {u"error": u"no model found with alias: default"},
            payload={u"q": u"food"}
        ),
    ]
    for test in tests:
        response = client.post(test.endpoint, data=json.dumps(test.payload), content_type='application/json')
        assert response.status_code == 200 and response.json == test.expected_response
