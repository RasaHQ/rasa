import sys

import json
import os
from aioresponses import aioresponses

from rasa.core.agent import Agent
from rasa.core.train import train
from rasa.core.utils import AvailableEndpoints
from rasa.utils.endpoints import EndpointConfig, ClientResponseError


async def test_moodbot_example(unpacked_trained_moodbot_path):
    agent = Agent.load(unpacked_trained_moodbot_path)

    responses = await agent.handle_text("/greet")
    assert responses[0]["text"] == "Hey! How are you?"

    responses.extend(await agent.handle_text("/mood_unhappy"))
    assert responses[-1]["text"] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


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
        policy_config="rasa/cli/default_config.yml",
    )
    response = {
        "events": [
            {"event": "form", "name": "restaurant_form", "timestamp": None},
            {
                "event": "slot",
                "timestamp": None,
                "name": "requested_slot",
                "value": "cuisine",
            },
        ],
        "responses": [{"template": "utter_ask_cuisine"}],
    }

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions", payload=response, repeat=True
        )

        responses = await agent.handle_text("/request_restaurant")

        assert responses[0]["text"] == "what cuisine?"

    response = {
        "error": "Failed to validate slot cuisine with action restaurant_form",
        "action_name": "restaurant_form",
    }

    with aioresponses() as mocked:
        # noinspection PyTypeChecker
        mocked.post(
            "https://example.com/webhooks/actions",
            repeat=True,
            exception=ClientResponseError(400, "", json.dumps(response)),
        )

        responses = await agent.handle_text("/chitchat")

        assert responses[0]["text"] == "chitchat"


async def test_restaurantbot_example():
    sys.path.append("examples/restaurantbot/")
    from run import train_core, train_nlu, parse

    p = "examples/restaurantbot/"
    stories = os.path.join("data", "test_stories", "stories_babi_small.md")
    nlu_data = os.path.join(p, "data", "nlu.md")
    await train_core(
        os.path.join(p, "domain.yml"), os.path.join(p, "models"), "current", stories
    )
    train_nlu(
        os.path.join(p, "config.yml"), os.path.join(p, "models"), "current", nlu_data
    )

    responses = await parse("hello", os.path.join(p, "models", "current"))

    assert responses[0]["text"] == "how can I help you?"
