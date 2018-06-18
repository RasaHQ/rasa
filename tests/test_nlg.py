from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import uuid

import pytest
from flask import Flask, request, jsonify
from pytest_localserver.http import WSGIServer

from rasa_core.actions.action import EndpointConfig
from rasa_core.agent import Agent


def nlg_app(base_url="/"):
    app = Flask(__name__)

    @app.route(base_url, methods=['POST'])
    def generate():
        """Check if the server is running and responds with the version."""
        nlg_call = request.json
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

    response = agent.handle_text("/greet", sender_id=sender)
    assert len(response) == 1
    assert response[0] == {"text": "Hey there!", "recipient_id": sender}
