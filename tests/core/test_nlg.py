import uuid
from typing import Text, Any

import jsonschema
import pytest
from sanic import Sanic, response

from rasa.core.nlg.callback import CallbackNaturalLanguageGenerator
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from rasa.shared.core.domain import Domain
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig, read_endpoint_config
from rasa.core.agent import Agent


def nlg_app(base_url="/"):

    app = Sanic(__name__)

    @app.route(base_url, methods=["POST"])
    async def generate(request):
        """Simple HTTP NLG generator, checks that the incoming request
        is format according to the spec."""

        nlg_request_format_spec = {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "arguments": {"type": "object"},
                "tracker": {
                    "type": "object",
                    "properties": {
                        "sender_id": {"type": "string"},
                        "slots": {"type": "object"},
                        "latest_message": {"type": "object"},
                        "latest_event_time": {"type": "number"},
                        "paused": {"type": "boolean"},
                        "events": {"type": "array"},
                    },
                },
                "channel": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }

        nlg_call = request.json

        jsonschema.validate(nlg_call, nlg_request_format_spec)

        if nlg_call.get("response") == "utter_greet":
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


def test_nlg_endpoint_config_loading(endpoints_path: Text):
    cfg = read_endpoint_config(endpoints_path, "nlg")

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


def test_nlg_schema_validation_empty_custom_dict():
    content = {"custom": {}}
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
def test_nlg_fill_response_text(slot_name: Text, slot_value: Any):
    response = {"text": f"{{{slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(response=response, filled_slots={slot_name: slot_value})
    assert result == {"text": str(slot_value)}


@pytest.mark.parametrize(
    "img_slot_name, img_slot_value",
    [("url", "https://www.exampleimg.com"), ("img1", "https://www.appleimg.com")],
)
def test_nlg_fill_response_image(img_slot_name: Text, img_slot_value: Text):
    response = {"image": f"{{{img_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response, filled_slots={img_slot_name: img_slot_value}
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
def test_nlg_fill_response_custom(slot_name: Text, slot_value: Any):
    response = {
        "custom": {
            "field": f"{{{slot_name}}}",
            "properties": {"field_prefixed": f"prefix_{{{slot_name}}}"},
            "bool_field": True,
            "int_field:": 42,
            "empty_field": None,
        }
    }
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(response=response, filled_slots={slot_name: slot_value})

    assert result == {
        "custom": {
            "field": str(slot_value),
            "properties": {"field_prefixed": f"prefix_{slot_value}"},
            "bool_field": True,
            "int_field:": 42,
            "empty_field": None,
        }
    }


def test_nlg_fill_response_custom_with_list():
    response = {
        "custom": {
            "blocks": [{"fields": [{"text": "*Departure date:*\n{test}"}]}],
            "other": ["{test}"],
        }
    }
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(response=response, filled_slots={"test": 5})
    assert result == {
        "custom": {
            "blocks": [{"fields": [{"text": "*Departure date:*\n5"}]}],
            "other": ["5"],
        }
    }


@pytest.mark.parametrize(
    "response_text, expected",
    [
        ('{{"variable":"{slot_1}"}}', '{"variable":"foo"}'),
        ("{slot_1} and {slot_2}", "foo and bar"),
        ("{{{slot_1}, {slot_2}!}}", "{foo, bar!}"),
        ("{{{slot_1}}}", "{foo}"),
        ("{{slot_1}}", "{slot_1}"),
    ],
)
def test_nlg_fill_response_text_with_json(response_text, expected):
    response = {"text": response_text}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response, filled_slots={"slot_1": "foo", "slot_2": "bar"}
    )
    assert result == {"text": expected}


@pytest.mark.parametrize("slot_name, slot_value", [("tag_w_\n", "a")])
def test_nlg_fill_response_with_bad_slot_name(slot_name, slot_value):
    response_text = f"{{{slot_name}}}"
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response={"text": response_text}, filled_slots={slot_name: slot_value}
    )
    assert result["text"] == response_text


@pytest.mark.parametrize(
    "text_slot_name, text_slot_value, img_slot_name, img_slot_value",
    [
        ("tag_w_underscore", "a", "url", "https://www.exampleimg.com"),
        ("tag with space", "bacon", "img1", "https://www.appleimg.com"),
    ],
)
def test_nlg_fill_response_image_and_text(
    text_slot_name, text_slot_value, img_slot_name, img_slot_value
):
    response = {"text": f"{{{text_slot_name}}}", "image": f"{{{img_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response,
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
def test_nlg_fill_response_text_and_custom(
    text_slot_name, text_slot_value, cust_slot_name, cust_slot_value
):
    response = {
        "text": f"{{{text_slot_name}}}",
        "custom": {
            "field": f"{{{cust_slot_name}}}",
            "properties": {"field_prefixed": f"prefix_{{{cust_slot_name}}}"},
        },
    }
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response,
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
def test_nlg_fill_response_attachment(attach_slot_name, attach_slot_value):
    response = {"attachment": "{" + attach_slot_name + "}"}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response, filled_slots={attach_slot_name: attach_slot_value}
    )
    assert result == {"attachment": str(attach_slot_value)}


@pytest.mark.parametrize(
    "button_slot_name, button_slot_value", [("button_1", "button_1")]
)
def test_nlg_fill_response_button(button_slot_name, button_slot_value):
    response = {
        "buttons": [
            {
                "payload": f'/choose{{{{"some_slot": "{{{button_slot_name}}}"}}}}',
                "title": f"{{{button_slot_name}}}",
            }
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response, filled_slots={button_slot_name: button_slot_value}
    )
    assert result == {
        "buttons": [
            {
                "payload": f'/choose{{"some_slot": "{button_slot_value}"}}',
                "title": f"{button_slot_value}",
            }
        ]
    }


@pytest.mark.parametrize(
    "quick_replies_slot_name, quick_replies_slot_value", [("qreply", "reply 1")]
)
def test_nlg_fill_response_quick_replies(
    quick_replies_slot_name, quick_replies_slot_value
):
    response = {"quick_replies": f"{{{quick_replies_slot_name}}}"}
    t = TemplatedNaturalLanguageGenerator(responses=dict())
    result = t._fill_response(
        response=response,
        filled_slots={quick_replies_slot_name: quick_replies_slot_value},
    )
    assert result == {"quick_replies": str(quick_replies_slot_value)}


async def test_nlg_conditional_response_variations_with_no_slots():
    responses = {
        "utter_test": [
            {
                "text": "Conditional OS Response A",
                "condition": [{"type": "slot", "name": "slot test", "value": "A"}],
                "channel": "os",
            },
            {
                "text": "Conditional Response A",
                "condition": [{"type": "slot", "name": "slot test", "value": "A"}],
            },
            {
                "text": "Conditional Response B",
                "condition": [{"type": "slot", "name": "slot test", "value": "B"}],
            },
            {"text": "Default response"},
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    no_slots_tracker = DialogueStateTracker(sender_id="nlg_test_default", slots=None)
    default_response = await t.generate(
        utter_action="utter_test", tracker=no_slots_tracker, output_channel=""
    )

    assert default_response.get("text") == "Default response"


async def test_nlg_when_multiple_conditions_satisfied():
    responses = {
        "utter_action": [
            {
                "text": "example A",
                "condition": [{"type": "slot", "name": "test", "value": "A"}],
            },
            {
                "text": "example B",
                "condition": [{"type": "slot", "name": "test_another", "value": "B"}],
            },
            {
                "text": "non matching example 1",
                "condition": [
                    {"type": "slot", "name": "test_third_slot", "value": "C"}
                ],
            },
            {
                "text": "non matching example 2",
                "condition": [{"type": "slot", "name": "test", "value": "D"}],
            },
        ]
    }

    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot_a = Slot(name="test", initial_value="A", influence_conversation=False)
    slot_b = Slot(name="test_another", initial_value="B", influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="test_nlg", slots=[slot_a, slot_b])
    resp = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert resp.get("text") in ["example A", "example B"]


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "response_variation"),
    (("test", "A", "example one A"), ("test", "B", "example two B")),
)
async def test_nlg_conditional_response_variations_with_interpolated_slots(
    slot_name, slot_value, response_variation
):
    responses = {
        "utter_action": [
            {
                "text": "example one {test}",
                "condition": [{"type": "slot", "name": "test", "value": "A"}],
            },
            {
                "text": "example two {test}",
                "condition": [{"type": "slot", "name": "test", "value": "B"}],
            },
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot = Slot(name=slot_name, initial_value=slot_value, influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="nlg_interpolated", slots=[slot])

    r = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert r.get("text") == response_variation


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "bot_message"),
    (
        (
            "can_withdraw",
            False,
            "You are not allowed to withdraw any amounts. Please check permission.",
        ),
        (
            "account_type",
            "secondary",
            "Withdrawal was sent for approval to primary account holder.",
        ),
    ),
)
async def test_nlg_conditional_response_variations_with_yaml_single_condition(
    slot_name, slot_value, bot_message
):
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    slot = Slot(name=slot_name, initial_value=slot_value, influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="conversation_id", slots=[slot])

    r = await t.generate(
        utter_action="utter_withdraw", tracker=tracker, output_channel=""
    )
    assert r.get("text") == bot_message


async def test_nlg_conditional_response_variations_with_yaml_multi_constraints():
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    first_slot = Slot(
        name="account_type", initial_value="primary", influence_conversation=False
    )
    second_slot = Slot(
        name="can_withdraw", initial_value=True, influence_conversation=False
    )
    tracker = DialogueStateTracker(
        sender_id="conversation_id", slots=[first_slot, second_slot]
    )
    r = await t.generate(
        utter_action="utter_withdraw", tracker=tracker, output_channel=""
    )
    assert r.get("text") == "Withdrawal has been approved."


async def test_nlg_conditional_response_variations_with_yaml_and_channel():
    domain = Domain.from_file(
        path="data/test_domains/conditional_response_variations.yml"
    )
    t = TemplatedNaturalLanguageGenerator(responses=domain.responses)

    slot = Slot(
        name="account_type", initial_value="primary", influence_conversation=False
    )
    tracker = DialogueStateTracker(sender_id="conversation_id", slots=[slot])

    r = await t.generate(
        utter_action="utter_check_balance", tracker=tracker, output_channel="os"
    )
    assert (
        r.get("text")
        == "As a primary account holder, you can now set-up your access on mobile app too."
    )

    resp = await t.generate(
        utter_action="utter_check_balance", tracker=tracker, output_channel="app"
    )
    assert resp.get("text") == "Welcome to your app account overview."


@pytest.mark.parametrize(
    ("slot_name", "slot_value", "message"),
    (
        ("test_bool", True, "example boolean"),
        ("test_int", 12, "example integer"),
        ("test_list", [], "example list"),
    ),
)
async def test_nlg_conditional_response_variations_with_diff_slot_types(
    slot_name, slot_value, message
):
    responses = {
        "utter_action": [
            {
                "text": "example boolean",
                "condition": [{"type": "slot", "name": "test_bool", "value": True}],
            },
            {
                "text": "example integer",
                "condition": [{"type": "slot", "name": "test_int", "value": 12}],
            },
            {
                "text": "example list",
                "condition": [{"type": "slot", "name": "test_list", "value": []}],
            },
        ]
    }
    t = TemplatedNaturalLanguageGenerator(responses=responses)
    slot = Slot(name=slot_name, initial_value=slot_value, influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="nlg_tracker", slots=[slot])

    r = await t.generate(
        utter_action="utter_action", tracker=tracker, output_channel=""
    )
    assert r.get("text") == message


async def test_nlg_non_matching_channel():
    domain = Domain.from_yaml(
        """
    responses:
        utter_hi:
        - text: "Hello"
        - text: "Hello Slack"
          channel: "slack"
    """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    tracker = DialogueStateTracker(sender_id="test", slots=[])
    r = await t.generate("utter_hi", tracker, "signal")
    assert r.get("text") == "Hello"


async def test_nlg_conditional_response_variations_with_none_slot():
    domain = Domain.from_yaml(
        """
        responses:
            utter_action:
            - text: "text A"
              condition:
              - type: slot
                name: account
                value: "A"
        """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = Slot(name="account", initial_value=None, influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r is None


async def test_nlg_conditional_response_variations_with_slot_not_a_constraint():
    domain = Domain.from_yaml(
        """
            responses:
                utter_action:
                - text: "text A"
                  condition:
                  - type: slot
                    name: account
                    value: "A"
            """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = Slot(name="account", initial_value="B", influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r is None


async def test_nlg_conditional_response_variations_with_null_slot():
    domain = Domain.from_yaml(
        """
                responses:
                    utter_action:
                    - text: "text for null"
                      condition:
                      - type: slot
                        name: account
                        value: null
                """
    )
    t = TemplatedNaturalLanguageGenerator(domain.responses)
    slot = Slot(name="account", initial_value=None, influence_conversation=False)
    tracker = DialogueStateTracker(sender_id="test", slots=[slot])
    r = await t.generate("utter_action", tracker, "")
    assert r.get("text") == "text for null"

    tracker_no_slots = DialogueStateTracker(sender_id="new_test", slots=[])
    r = await t.generate("utter_action", tracker_no_slots, "")
    assert r.get("text") == "text for null"
