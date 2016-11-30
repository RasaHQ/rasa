import pytest
import requests
import os
from rasa_nlu.server import RasaNLUServer
import threading
import time
import json

@pytest.fixture
def http_server():
    def url(port):
        return "http://localhost:%s" % (port)
    # basic conf
    config = {
        'logfile': os.path.join(os.getcwd(), "rasa_nlu_logs.json"),
        'port': 5001,
        "backend": "mitie",
        "path" : "./",
        "data" : "./data/demo-restaurants.json",
        "emulate": "wit"
    }
    # run server in background
    server = RasaNLUServer(config)
    t = threading.Thread(target=server.start)
    t.setDaemon(True)
    t.start()
    # TODO: implement better way to notify when server is up
    time.sleep(2)
    port = server.server.server_address[1]
    yield url(port)
    server.stop()

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
    assert req.status_code == 200 and req.json() == [{"entities": {}, "confidence": None, "intent": "greet", "_text": "hello"}]

def test_post_parse(http_server):
    req = requests.post(http_server + "/parse", json={"q": "hello"})
    assert req.status_code == 200 and req.json() == [{"entities": {}, "confidence": None, "intent": "greet", "_text": "hello"}]
    
