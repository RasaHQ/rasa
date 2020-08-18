import sys

import json
from pathlib import Path
from typing import Text

from aioresponses import aioresponses

from rasa.core.agent import Agent
from rasa.core.train import train
from rasa.core.utils import AvailableEndpoints
from rasa.importers.importer import TrainingDataImporter
from rasa.utils.endpoints import EndpointConfig, ClientResponseError


async def test_moodbot_example(unpacked_trained_moodbot_path: Text):
    agent = Agent.load(unpacked_trained_moodbot_path)

    responses = await agent.handle_text("/greet")
    assert responses[0]["text"] == "Hey! How are you?"

    responses.extend(await agent.handle_text("/mood_unhappy"))
    assert responses[-1]["text"] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4


async def test_formbot_example():
    sys.path.append("examples/formbot/")
    project = Path("examples/formbot/")
    config = str(project / "config.yml")
    domain = str(project / "domain.yml")
    training_dir = project / "data"
    training_files = [
        str(training_dir / "rules.yml"),
        str(training_dir / "stories.yml"),
    ]
    importer = TrainingDataImporter.load_from_config(config, domain, training_files)

    endpoint = EndpointConfig("https://example.com/webhooks/actions")
    endpoints = AvailableEndpoints(action=endpoint)
    agent = await train(
        domain,
        importer,
        str(project / "models" / "dialogue"),
        endpoints=endpoints,
        policy_config="examples/formbot/config.yml",
    )

    async def mock_form_happy_path(input_text, output_text, slot=None):
        if slot:
            form = "restaurant_form"
            template = f"utter_ask_{slot}"
        else:
            form = None
            template = "utter_submit"
        response = {
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
        with aioresponses() as mocked:
            mocked.post(
                "https://example.com/webhooks/actions", payload=response, repeat=True
            )
            responses = await agent.handle_text(input_text)
            assert responses[0]["text"] == output_text

    async def mock_form_unhappy_path(input_text, output_text, slot):
        response_error = {
            "error": f"Failed to extract slot {slot} with action restaurant_form",
            "action_name": "restaurant_form",
        }
        with aioresponses() as mocked:
            # noinspection PyTypeChecker
            mocked.post(
                "https://example.com/webhooks/actions",
                repeat=True,
                exception=ClientResponseError(400, "", json.dumps(response_error)),
            )
            responses = await agent.handle_text(input_text)
            assert responses[0]["text"] == output_text

    await mock_form_happy_path("/request_restaurant", "what cuisine?", slot="cuisine")
    await mock_form_unhappy_path("/chitchat", "chitchat", slot="cuisine")
    await mock_form_happy_path(
        '/inform{"cuisine": "mexican"}', "how many people?", slot="num_people"
    )
    await mock_form_happy_path(
        '/inform{"number": "2"}', "do you want to seat outside?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "please provide additional preferences", slot="preferences"
    )

    responses = await agent.handle_text("/restart")
    assert responses[0]["text"] == "restarted"

    responses = await agent.handle_text("/greet")
    assert (
        responses[0]["text"]
        == "Hello! I am restaurant search assistant! How can I help?"
    )

    await mock_form_happy_path("/request_restaurant", "what cuisine?", slot="cuisine")
    await mock_form_happy_path(
        '/inform{"cuisine": "mexican"}', "how many people?", slot="num_people"
    )
    await mock_form_happy_path(
        '/inform{"number": "2"}', "do you want to seat outside?", slot="outdoor_seating"
    )
    await mock_form_unhappy_path(
        "/stop", "do you want to continue?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "do you want to seat outside?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "please provide additional preferences", slot="preferences"
    )
    await mock_form_happy_path(
        "/deny", "please give your feedback on your experience so far", slot="feedback"
    )
    await mock_form_happy_path('/inform{"feedback": "great"}', "All done!")

    responses = await agent.handle_text("/thankyou")
    assert responses[0]["text"] == "you are welcome :)"
