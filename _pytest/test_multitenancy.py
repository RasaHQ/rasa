# -*- coding: utf-8 -*-

import pytest
import requests
import os
from rasa_nlu.server import RasaNLUServer
from rasa_nlu.config import RasaNLUConfig
from multiprocessing import Process
import time
import json
import codecs


class ResponseTest():
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload


@pytest.fixture
def http_server():
    def url(port):
        return "http://localhost:{0}".format(port)
    if "TRAVIS_BUILD_DIR" in os.environ:
        wd = os.environ["TRAVIS_BUILD_DIR"]
        print("travis dir: {0}".format(wd))
        root_dir = os.path.join(wd, '/')
        print("{0} exists {1}".format(root_dir,os.path.exists(root_dir)))
        model_dir = os.path.join(root_dir, 'models/model_1')
        print("{0} exists {1}".format(model_dir,os.path.exists(model_dir)))
    else:
        root_dir = os.getcwd()
        print("model at")
        print(os.path.join(root_dir, "models/model_1"))
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
        "backend": "mitie",
        "path": root_dir,
        "data": os.path.join(root_dir, "data/demo-restaurants.json"),
        "server_model_dir": {
          "one": os.path.join(root_dir, "models/model_1"),
          "two": os.path.join(root_dir, "models/model_2")
        }
    }
    config = RasaNLUConfig(cmdline_args=_config)
    # run server in background
    server = RasaNLUServer(config)
    p = Process(target=server.start)
    p.daemon = True
    p.start()
    # TODO: implement better way to notify when server is up
    time.sleep(2)
    yield url(config.port)
    p.terminate()


def test_get_parse(http_server):
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
        req = requests.get(http_server + test.endpoint)
        assert req.status_code == 200 and req.json() == test.expected_response


def test_post_parse(http_server):
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
        req = requests.post(http_server + test.endpoint, json=test.payload)
        assert req.status_code == 200 and req.json() == test.expected_response
