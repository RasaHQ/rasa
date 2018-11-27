import os
import sys
import json
from httpretty import httpretty

from rasa_core.train import train_dialogue_model
from rasa_core.agent import Agent
from rasa_core.utils import EndpointConfig, AvailableEndpoints


def test_moodbot_example(trained_moodbot_path):
    agent = Agent.load(trained_moodbot_path)

    responses = agent.handle_text("/greet")
    assert responses[0]['text'] == 'Hey! How are you?'

    responses.extend(agent.handle_text("/mood_unhappy"))
    assert responses[-1]['text'] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from bot import train_dialogue

    p = "examples/restaurantbot/"
    stories = os.path.join("data", "test_stories", "stories_babi_small.md")
    agent = train_dialogue(os.path.join(p, "restaurant_domain.yml"),
                           os.path.join(p, "models", "dialogue"),
                           stories)

    responses = agent.handle_text("/greet")
    assert responses[0]['text'] == 'how can I help you?'


def test_formbot_example():
    sys.path.append("examples/formbot/")

    p = "examples/formbot/"
    stories = os.path.join(p, "data", "stories.md")
    endpoint = EndpointConfig("https://abc.defg/webhooks/actions")
    endpoints = AvailableEndpoints(action=endpoint)
    agent = train_dialogue_model(os.path.join(p, "domain.yml"),
                                 stories,
                                 os.path.join(p, "models", "dialogue"),
                                 endpoints=endpoints,
                                 policy_config="rasa_core/default_config.yml")
    response = {
        'events': [
            {'event': 'form', 'name': 'restaurant_form', 'timestamp': None},
            {'event': 'slot', 'timestamp': None,
             'name': 'requested_slot', 'value': 'cuisine'}
        ],
        'responses': [
            {'template': 'utter_ask_cuisine'}
        ]
    }

    httpretty.register_uri(
        httpretty.POST,
        'https://abc.defg/webhooks/actions',
        body=json.dumps(response))

    httpretty.enable()

    responses = agent.handle_text("/request_restaurant")

    httpretty.disable()

    assert responses[0]['text'] == 'what cuisine?'

    response = {
        "error": "Failed to validate slot cuisine with action restaurant_form",
        "action_name": "restaurant_form"
    }

    httpretty.register_uri(
        httpretty.POST,
        'https://abc.defg/webhooks/actions',
        status=400,
        body=json.dumps(response))

    httpretty.enable()

    responses = agent.handle_text("/chitchat")

    httpretty.disable()

    assert responses[0]['text'] == 'chitchat'
