# -*- coding: utf-8 -*-
import tempfile

import pytest
import requests
import os
from rasa_nlu.server import RasaNLUServer
from rasa_nlu.config import RasaNLUConfig
from multiprocessing import Process
import time
import json
import codecs


class ResponseTest(object):
    def __init__(self, endpoint, expected_response, payload=None):
        self.endpoint = endpoint
        self.expected_response = expected_response
        self.payload = payload


@pytest.fixture
def http_server(port_getter):
    def url(port):
        return "http://localhost:{0}".format(port)
    # basic conf
    _, nlu_log_file = tempfile.mkstemp(suffix="_rasa_nlu_logs.json")
    _config = {
        'write': nlu_log_file,
        'port': port_getter(),
        "backend": "mitie",
        "path": "./",
        "data": "./data/demo-restaurants.json",
        "emulate": "wit"
    }
    config = RasaNLUConfig(cmdline_args=_config)
    # run server in background
    server = RasaNLUServer(config)
    p = Process(target=server.start)
    p.daemon = True
    p.start()
    # TODO: implement better way to notify when server is up
    time.sleep(2)
    yield url(config['port'])
    p.terminate()
    os.remove(nlu_log_file)


def test_root(http_server):
    req = requests.get(http_server)
    ret = req.text

    assert req.status_code == 200 and ret == "hello"


def test_status(http_server):
    req = requests.get(http_server + "/status")
    ret = req.json()
    assert req.status_code == 200 and ("training" in ret and "available_models" in ret)


def test_get_parse(http_server):
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
        req = requests.get(http_server + test.endpoint)
        assert req.status_code == 200 and req.json() == test.expected_response


def test_post_parse(http_server):
    tests = [
        ResponseTest(
            u"/parse",
            [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello"}],
            payload={u"q": u"hello"}
        ),
        ResponseTest(
            u"/parse",
            [{u"entities": {}, u"confidence": 1.0, u"intent": u"greet", u"_text": u"hello ńöñàśçií"}],
            payload={u"q": u"hello ńöñàśçií"}
        ),
    ]
    for test in tests:
        req = requests.post(http_server + test.endpoint, json=test.payload)
        assert req.status_code == 200 and req.json() == test.expected_response


def test_post_train(http_server):
    train_data = json.loads(codecs.open('data/examples/luis/demo-restaurants.json',
                                        encoding='utf-8').read())
    req = requests.post(http_server + "/parse", json=train_data)
    assert req.status_code == 200
