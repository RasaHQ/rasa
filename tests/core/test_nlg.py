import uuid
from typing import Text, Any

import jsonschema
import pytest
from sanic import Sanic, response

from rasa.core.nlg.callback import (
    nlg_request_format_spec,
    CallbackNaturalLanguageGenerator,
)
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from rasa.core.agent import Agent
from tests.core.conftest import DEFAULT_ENDPOINTS_FILE


def nlg_app(base_url="/"):

    app = Sanic(__name__)

    @app.route(base_url, methods=["POST"])
    async def generate(request):
        """Simple HTTP NLG generator, checks that the incoming request
        is format according to the spec."""

        nlg_call = request.json

        jsonschema.validate(nlg_call, nlg_request_format_spec())

        if nlg_call.get("template") == "utter_greet":
            response_dict = {"text": "Hey there!"}
        else:
            response_dict = {"text": "Sorry, didn't get that."}
        return response.json(response_dict)

    return app


# noinspection PyShadowingNames
@pytest.fixture()
def http_nlg(loop, sanic_client):
    return loop.run_until_complete(sanic_client(nlg_app()))


async def test_nlg(http_nlg, trained_rasa_model):
    sender = str(uuid.uuid1())

    nlg_endpoint = EndpointConfig.from_dict({"url": http_nlg.make_url("/")})
    agent = Agent.load(trained_rasa_model, None, generator=nlg_endpoint)

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
        ("empty_string", ""),
        ("null", None),
    ],
)
def test_nlg_fill_template_text(slot_name: Text, slot_value: Any):
    template = {"text": f"{{{slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(template=template, filled_slots={slot_name: slot_value})
    assert result == {"text": str(slot_value)}


@pytest.mark.parametrize(
    "img_slot_name, img_slot_value",
    [("url", "https://www.exampleimg.com"), ("img1", "https://www.appleimg.com")],
)
def test_nlg_fill_template_image(img_slot_name: Text, img_slot_value: Text):
    template = {"image": f"{{{img_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template, filled_slots={img_slot_name: img_slot_value}
    )
    assert result == {"image": str(img_slot_value)}


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
        ("empty_string", ""),
        ("null", None),
    ],
)
def test_nlg_fill_template_custom(slot_name: Text, slot_value: Any):
    template = {
        "custom": {
            "field": f"{{{slot_name}}}",
            "properties": {"field_prefixed": f"prefix_{{{slot_name}}}"},
        }
    }
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(template=template, filled_slots={slot_name: slot_value})

    assert result == {
        "custom": {
            "field": str(slot_value),
            "properties": {"field_prefixed": f"prefix_{str(slot_value)}"},
        }
    }


def test_nlg_fill_template_custom_with_list():
    template = {
        "custom": {
            "blocks": [{"fields": [{"text": "*Departure date:*\n{test}"}]}],
            "other": ["{test}"],
        }
    }
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(template=template, filled_slots={"test": 5})
    assert result == {
        "custom": {
            "blocks": [{"fields": [{"text": "*Departure date:*\n5"}]}],
            "other": ["5"],
        }
    }


@pytest.mark.parametrize(
    "template_text, expected",
    [
        ('{{"variable":"{slot_1}"}}', '{"variable":"foo"}'),
        ("{slot_1} and {slot_2}", "foo and bar"),
        ("{{{slot_1}, {slot_2}!}}", "{foo, bar!}"),
        ("{{{slot_1}}}", "{foo}"),
        ("{{slot_1}}", "{slot_1}"),
    ],
)
def test_nlg_fill_template_text_with_json(template_text, expected):
    template = {"text": template_text}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template, filled_slots={"slot_1": "foo", "slot_2": "bar"}
    )
    assert result == {"text": expected}


@pytest.mark.parametrize("slot_name, slot_value", [("tag_w_\n", "a")])
def test_nlg_fill_template_with_bad_slot_name(slot_name, slot_value):
    template_text = f"{{{slot_name}}}"
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template={"text": template_text}, filled_slots={slot_name: slot_value}
    )
    assert result["text"] == template_text


@pytest.mark.parametrize(
    "text_slot_name, text_slot_value, img_slot_name, img_slot_value",
    [
        ("tag_w_underscore", "a", "url", "https://www.exampleimg.com"),
        ("tag with space", "bacon", "img1", "https://www.appleimg.com"),
    ],
)
def test_nlg_fill_template_image_and_text(
    text_slot_name, text_slot_value, img_slot_name, img_slot_value
):
    template = {"text": f"{{{text_slot_name}}}", "image": f"{{{img_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template,
        filled_slots={text_slot_name: text_slot_value, img_slot_name: img_slot_value},
    )
    assert result == {"text": str(text_slot_value), "image": str(img_slot_value)}


@pytest.mark.parametrize(
    "text_slot_name, text_slot_value, cust_slot_name, cust_slot_value",
    [
        ("tag_w_underscore", "a", "tag.with.dot", "chocolate"),
        ("tag with space", "bacon", "tag-w-dash", "apple pie"),
    ],
)
def test_nlg_fill_template_text_and_custom(
    text_slot_name, text_slot_value, cust_slot_name, cust_slot_value
):
    template = {
        "text": f"{{{text_slot_name}}}",
        "custom": {
            "field": f"{{{cust_slot_name}}}",
            "properties": {"field_prefixed": f"prefix_{{{cust_slot_name}}}"},
        },
    }
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template,
        filled_slots={text_slot_name: text_slot_value, cust_slot_name: cust_slot_value},
    )
    assert result == {
        "text": str(text_slot_value),
        "custom": {
            "field": str(cust_slot_value),
            "properties": {"field_prefixed": f"prefix_{str(cust_slot_value)}"},
        },
    }


@pytest.mark.parametrize(
    "attach_slot_name, attach_slot_value", [("attach_file", "https://attach.pdf")]
)
def test_nlg_fill_template_attachment(attach_slot_name, attach_slot_value):
    template = {"attachment": "{" + attach_slot_name + "}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template, filled_slots={attach_slot_name: attach_slot_value}
    )
    assert result == {"attachment": str(attach_slot_value)}


@pytest.mark.parametrize(
    "button_slot_name, button_slot_value", [("button_1", "button1")]
)
def test_nlg_fill_template_button(button_slot_name, button_slot_value):
    template = {"button": f"{{{button_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template, filled_slots={button_slot_name: button_slot_value}
    )
    assert result == {"button": str(button_slot_value)}


@pytest.mark.parametrize(
    "quick_replies_slot_name, quick_replies_slot_value", [("qreply", "reply 1")]
)
def test_nlg_fill_template_quick_replies(
    quick_replies_slot_name, quick_replies_slot_value
):
    template = {"quick_replies": f"{{{quick_replies_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(templates=dict())
    result = t._fill_template(
        template=template,
        filled_slots={quick_replies_slot_name: quick_replies_slot_value},
    )
    assert result == {"quick_replies": str(quick_replies_slot_value)}
