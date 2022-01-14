from typing import Type

import pytest
from jsonschema import ValidationError, validate

from rasa.core.actions.action import RemoteAction
from rasa.shared.core.events import Event
from rasa.shared.utils.schemas.events import EVENT_SCHEMA
import rasa.shared.utils.common

TEST_EVENT = {
    "sender_id": "77bd4d3841294f1f9f82ef8cfe9321a7",
    "event": "user",
    "timestamp": 1640019570.8848355,
    "text": "hey 👋🏻",
    "parse_data": {
        "intent": {
            "id": 5103161883140543062,
            "name": "greet",
            "confidence": 0.9999818801879883,
        },
        "entities": [],
        "text": "hey",
        "message_id": "52ec1cba47c840d798d9f612334a750d",
        "metadata": {},
        "intent_ranking": [
            {
                "id": 5103161883140543062,
                "name": "greet",
                "confidence": 0.9999818801879883,
            }
        ],
        "response_selector": {
            "all_retrieval_intents": ["chitchat"],
            "faq": {
                "response": {
                    "id": None,
                    "responses": None,
                    "response_templates": None,
                    "confidence": 0.0,
                    "intent_response_key": None,
                    "utter_action": "utter_None",
                    "template_name": "utter_None",
                },
                "ranking": [],
            },
            "chitchat": {
                "response": {
                    "id": -2223917631873543698,
                    "responses": [{"text": "I am called Retrieval Bot!"}],
                    "response_templates": [{"text": "I am called Retrieval Bot!"}],
                    "confidence": 0.9660752415657043,
                    "intent_response_key": "chitchat/ask_name",
                    "utter_action": "utter_chitchat/ask_name",
                    "template_name": "utter_chitchat/ask_name",
                },
                "ranking": [
                    {
                        "id": -2223917631873543698,
                        "confidence": 0.9660752415657043,
                        "intent_response_key": "chitchat/ask_name",
                    }
                ],
            },
        },
    },
    "input_channel": "rasa",
    "message_id": "52ec1cba47c840d798d9f612334a750d",
    "metadata": {},
}


@pytest.mark.parametrize("event_class", rasa.shared.utils.common.all_subclasses(Event))
def test_remote_action_validate_all_event_subclasses(event_class: Type[Event]):

    if event_class.type_name == "slot":
        response = {
            "events": [{"event": "slot", "name": "test", "value": "example"}],
            "responses": [],
        }
    elif event_class.type_name == "entities":
        response = {"events": [{"event": "entities", "entities": []}], "responses": []}
    else:
        response = {"events": [{"event": event_class.type_name}], "responses": []}

    # ignore the below events since these are not sent or received outside Rasa
    if event_class.type_name not in [
        "wrong_utterance",
        "wrong_action",
        "warning_predicted",
    ]:
        validate(response, RemoteAction.action_response_format_spec())


def test_validate_single_event():
    validate(TEST_EVENT, EVENT_SCHEMA)


def test_validate_single_event_raises():
    test_wrong_schema = TEST_EVENT.copy()
    test_wrong_schema["event"] = "non-existing event 🪲"

    with pytest.raises(ValidationError):
        validate(test_wrong_schema, EVENT_SCHEMA)
