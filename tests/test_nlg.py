import uuid

import jsonschema
import pytest
from flask import Flask, request, jsonify
from pytest_localserver.http import WSGIServer

from rasa_core import utils
from rasa_core.nlg.callback import (
    nlg_request_format_spec,
    CallbackNaturalLanguageGenerator)
from rasa_core.utils import EndpointConfig
from rasa_core.agent import Agent
from tests.conftest import DEFAULT_ENDPOINTS_FILE


def nlg_app(base_url="/"):
    app = Flask(__name__)

    @app.route(base_url, methods=['POST'])
    def generate():
        """Simple HTTP NLG generator, checks that the incoming request
        is format according to the spec."""

        nlg_call = request.json

        jsonschema.validate(nlg_call, nlg_request_format_spec())

        if nlg_call.get("template") == "utter_greet":
            response = {"text": "Hey there!"}
        else:
            response = {"text": "Sorry, didn't get that."}
        return jsonify(response)

    return app


@pytest.fixture(scope="module")
def http_nlg(request):
    http_server = WSGIServer(application=nlg_app())
    http_server.start()

    request.addfinalizer(http_server.stop)
    return http_server.url


def test_nlg(http_nlg, default_agent_path):
    sender = str(uuid.uuid1())

    nlg_endpoint = EndpointConfig.from_dict({
        "url": http_nlg
    })
    agent = Agent.load(default_agent_path, None,
                       generator=nlg_endpoint)

    response = agent.handle_message("/greet", sender_id=sender)
    assert len(response) == 1
    assert response[0] == {"text": "Hey there!", "recipient_id": sender}


def test_nlg_endpoint_config_loading():
    cfg = utils.read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "nlg")

    assert cfg == EndpointConfig.from_dict({
        "url": "http://localhost:5055/nlg"
    })


def test_nlg_schema_validation():
    content = {"text": "Hey there!"}
    assert CallbackNaturalLanguageGenerator.validate_response(content)


def test_nlg_schema_validation_empty_buttons():
    content = {"text": "Hey there!", "buttons": []}
    assert CallbackNaturalLanguageGenerator.validate_response(content)


def test_nlg_schema_validation_empty_image():
    content = {"text": "Hey there!", "image": None}
    assert CallbackNaturalLanguageGenerator.validate_response(content)
