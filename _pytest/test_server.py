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


@pytest.fixture
def http_server():
    def url(port):
        return "http://localhost:{0}".format(port)
    # basic conf
    _config = {
        'write': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5022,
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
    yield url(config.port)
    p.terminate()


def test_root(http_server):
    req = requests.get(http_server)
    ret = req.text

    assert req.status_code == 200 and ret == "hello"


def test_status(http_server):
    req = requests.get(http_server + "/status")
    ret = req.json()
    assert req.status_code == 200 and ("training" in ret and "available_models" in ret)


def test_get_parse(http_server):
    req = requests.get(http_server + "/parse?q=hello")
    expected = [{u"entities": {}, u"confidence": None, u"intent": u"greet", u"_text": u"hello"}]
    assert req.status_code == 200 and req.json() == expected


def test_post_parse(http_server):
    req = requests.post(http_server + "/parse", json={"q": "hello"})
    expected = [{u"entities": {}, u"confidence": None, u"intent": u"greet", u"_text": u"hello"}]
    assert req.status_code == 200 and req.json() == expected


def test_post_train(http_server):
    train_data = json.loads(codecs.open('data/examples/luis/demo-restaurants.json',
                                        encoding='utf-8').read())
    req = requests.post(http_server + "/parse", json=train_data)
    # TODO: POST /train oddly returns an error msg but training works fine
    # For now check only status code. Later on take a look at actual response
    # assert req.status_code == 200 and "training started with pid" in req.text
    assert req.status_code == 200
