import aiohttp
import sys

import json
import os
import pytest
from aioresponses import aioresponses

from rasa.core import utils
from rasa.core.agent import Agent
from rasa.core.train import train
from rasa.core.utils import (
    EndpointConfig, AvailableEndpoints,
    ClientResponseError)


@pytest.fixture(scope="session")
def loop():
    from pytest_sanic.plugin import loop as sanic_loop
    return utils.enable_async_loop_debugging(next(sanic_loop()))


async def test_moodbot_example(trained_moodbot_path):
    agent = Agent.load(trained_moodbot_path)

    responses = await agent.handle_text("/greet")
    assert responses[0]['text'] == 'Hey! How are you?'

    responses.extend(
        await agent.handle_text("/mood_unhappy"))
    assert responses[-1]['text'] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


async def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from bot import train_dialogue

    p = "examples/restaurantbot/"
    stories = os.path.join("data", "test_stories", "stories_babi_small.md")
    agent = await train_dialogue(os.path.join(p, "domain.yml"),
                                 os.path.join(p, "models", "dialogue"),
                                 stories)

    responses = await agent.handle_text("/greet")
    assert responses[0]['text'] == 'how can I help you?'


async def test_formbot_example():
    sys.path.append("examples/formbot/")

    p = "examples/formbot/"
    stories = os.path.join(p, "data", "stories.md")
    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    endpoints = AvailableEndpoints(action=endpoint)
    agent = await train(
        os.path.join(p, "domain.yml"),
        stories,
        os.path.join(p, "models", "dialogue"),
        endpoints=endpoints,
        policy_config="rasa/core/default_config.yml")
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

    with aioresponses() as mocked:
        mocked.post('https://example.com/webhooks/actions',
                    payload=response,
                    repeat=True)

        responses = await agent.handle_text("/request_restaurant")

        assert responses[0]['text'] == 'what cuisine?'

    response = {
        "error": "Failed to validate slot cuisine with action "
                 "restaurant_form",
        "action_name": "restaurant_form"
    }

    with aioresponses() as mocked:
        # noinspection PyTypeChecker
        mocked.post('https://example.com/webhooks/actions',
                    repeat=True,
                    exception=ClientResponseError(
                        400, "", json.dumps(response)))

        responses = await agent.handle_text("/chitchat")

        assert responses[0]['text'] == 'chitchat'


async def test_concertbot_training():
    from examples.concertbot.train import train_dialogue

    r = await train_dialogue(domain_file='examples/concertbot/domain.yml',
                             stories_file='examples/concertbot/data/stories.md',
                             model_path='examples/concertbot/models/dialogue',
                             policy_config='examples/concertbot/'
                                           'config.yml')
    assert r is not None
