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
        policy_config="examples/formbot/config.yml",
    )

    def response_slot(slot):
        if slot:
            form = "restaurant_form"
            template = f"utter_ask_{slot}"
        else:
            form = None
            template = "utter_submit"
        return {
            "events": [
                {"event": "form", "name": form, "timestamp": None},
                {
                    "event": "slot",
                    "timestamp": None,
                    "name": "requested_slot",
                    "value": slot,
                },
            ],
            "responses": [{"template": template}],
        }

    def response_error(slot):
        return {
            "error": f"Failed to extract slot {slot} with action restaurant_form",
            "action_name": "restaurant_form",
        }

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("cuisine"),
            repeat=True,
        )
        responses = await agent.handle_text("/request_restaurant")
        assert responses[0]["text"] == "what cuisine?"

    with aioresponses() as mocked:
        # noinspection PyTypeChecker
        mocked.post(
            "https://example.com/webhooks/actions",
            repeat=True,
            exception=ClientResponseError(
                400, "", json.dumps(response_error("cuisine"))
            ),
        )
        responses = await agent.handle_text("/chitchat")
        assert responses[0]["text"] == "chitchat"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("num_people"),
            repeat=True,
        )
        responses = await agent.handle_text('/inform{"cuisine": "mexican"}')
        assert responses[0]["text"] == "how many people?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("outdoor_seating"),
            repeat=True,
        )
        responses = await agent.handle_text('/inform{"number": "2"}')
        assert responses[0]["text"] == "do you want to seat outside?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("preferences"),
            repeat=True,
        )
        responses = await agent.handle_text("/affirm")
        assert responses[0]["text"] == "please provide additional preferences"

    responses = await agent.handle_text("/restart")
    assert responses[0]["text"] == "restarted"

    responses = await agent.handle_text("/greet")
    assert (
        responses[0]["text"]
        == "Hello! I am restaurant search assistant! How can I help?"
    )

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("cuisine"),
            repeat=True,
        )
        responses = await agent.handle_text("/request_restaurant")
        assert responses[0]["text"] == "what cuisine?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("num_people"),
            repeat=True,
        )
        responses = await agent.handle_text('/inform{"cuisine": "mexican"}')
        assert responses[0]["text"] == "how many people?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("outdoor_seating"),
            repeat=True,
        )
        responses = await agent.handle_text('/inform{"number": "2"}')
        assert responses[0]["text"] == "do you want to seat outside?"

    with aioresponses() as mocked:
        # noinspection PyTypeChecker
        mocked.post(
            "https://example.com/webhooks/actions",
            repeat=True,
            exception=ClientResponseError(
                400, "", json.dumps(response_error("outdoor_seating"))
            ),
        )
        responses = await agent.handle_text("/stop")
        assert responses[0]["text"] == "do you want to continue?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("outdoor_seating"),
            repeat=True,
        )
        responses = await agent.handle_text("/affirm")
        assert responses[0]["text"] == "do you want to seat outside?"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("preferences"),
            repeat=True,
        )
        responses = await agent.handle_text("/affirm")
        assert responses[0]["text"] == "please provide additional preferences"

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot("feedback"),
            repeat=True,
        )
        responses = await agent.handle_text("/deny")
        assert (
            responses[0]["text"]
            == "please give your feedback on your experience so far"
        )

    with aioresponses() as mocked:
        mocked.post(
            "https://example.com/webhooks/actions",
            payload=response_slot(None),
            repeat=True,
        )
        responses = await agent.handle_text('/inform{"feedback": "great"}')
        assert responses[0]["text"] == "All done!"

    responses = await agent.handle_text("/thankyou")
    assert responses[0]["text"] == "you are welcome :)"


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
