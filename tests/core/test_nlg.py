import asyncio
import uuid

import jsonschema
import pytest
from flask import Flask, request, jsonify
from pytest_localserver.http import WSGIServer

import rasa.utils.io
from rasa.core.nlg.callback import (
    nlg_request_format_spec,
    CallbackNaturalLanguageGenerator,
)
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from rasa.core.agent import Agent
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE


@pytest.fixture(scope="module")
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = rasa.utils.io.enable_async_loop_debugging(loop)
    yield loop
    loop.close()


def nlg_app(base_url="/"):
    app = Flask(__name__)

    @app.route(base_url, methods=["POST"])
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


# noinspection PyShadowingNames
@pytest.fixture(scope="module")
def http_nlg(request):
    http_server = WSGIServer(application=nlg_app())
    http_server.start()

    request.addfinalizer(http_server.stop)
    return http_server.url


async def test_nlg(http_nlg, default_agent_path):
    sender = str(uuid.uuid1())

    nlg_endpoint = EndpointConfig.from_dict({"url": http_nlg})
    agent = Agent.load(default_agent_path, None, generator=nlg_endpoint)

    response = await agent.handle_text("/greet", sender_id=sender)
    assert len(response) == 1
    assert response[0] == {"text": "Hey there!", "recipient_id": sender}


def test_nlg_endpoint_config_loading():
    cfg = read_endpoint_config(DEFAULT_ENDPOINTS_FILE, "nlg")

    assert cfg == EndpointConfig.from_dict({"url": "http://localhost:5055/nlg"})


def test_nlg_schema_validation():
    content = {"text": "Hey there!"}
    assert CallbackNaturalLanguageGenerator.validate_response(content)


def test_nlg_schema_validation_empty_buttons():
    content = {"text": "Hey there!", "buttons": []}
    assert CallbackNaturalLanguageGenerator.validate_response(content)


def test_nlg_schema_validation_empty_image():
    content = {"text": "Hey there!", "image": None}
    assert CallbackNaturalLanguageGenerator.validate_response(content)


@pytest.mark.parametrize(
    "slot_name, slot_value",
    [
        ("tag_w_underscore", "a"),
        ("tag with space", "bacon"),
        ("tag.with.dot", "chocolate"),
        ("tag-w-dash", "apple pie"),
        ("tag-w-$", "banana"),
        ("tag-w-@", "one"),
        ("tagCamelCase", "two"),
        ("tag-w-*", "three"),
        ("tag_w_underscore", "a"),
        ("tag.with.float.val", 1.3),
        ("tag-w-$", "banana"),
        ("tagCamelCase", "two"),
    ],
)
def test_nlg_fill_template_text(slot_name, slot_value):
    template = {"text": "{" + slot_name + "}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template_text(
        template=template, filled_slots={slot_name: slot_value}
    )
    assert result == {"text": str(slot_value)}


@pytest.mark.parametrize("slot_name, slot_value", [("tag_w_\n", "a")])
def test_nlg_fill_template_text_w_bad_slot_name2(slot_name, slot_value):
    template_text = "{" + slot_name + "}"
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template_text(
        template={"text": template_text}, filled_slots={slot_name: slot_value}
    )
    assert result["text"] == template_text
