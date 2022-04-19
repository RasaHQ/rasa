import json
from typing import Text, Optional, Dict, Any

import pytest
from aioresponses import aioresponses

from rasa.core.agent import Agent
from rasa.shared.core.domain import Domain
from rasa.utils.endpoints import ClientResponseError


@pytest.mark.timeout(300, func_only=True)
async def test_moodbot_example(trained_moodbot_path: Text):
    agent = Agent.load(trained_moodbot_path)

    responses = await agent.handle_text("/greet")
    assert responses[0]["text"] == "Hey! How are you?"

    responses.extend(await agent.handle_text("/mood_unhappy"))
    assert responses[-1]["text"] in {"Did that help you?"}

    # (there is a 'I am on it' message in the middle we are not checking)
    assert len(responses) == 4

    moodbot_domain = Domain.load("data/test_moodbot/domain.yml")
    assert agent.domain.action_names_or_texts == moodbot_domain.action_names_or_texts
    assert agent.domain.intents == moodbot_domain.intents
    assert agent.domain.entities == moodbot_domain.entities
    assert agent.domain.responses == moodbot_domain.responses
    assert [s.name for s in agent.domain.slots] == [
        s.name for s in moodbot_domain.slots
    ]


@pytest.mark.timeout(300, func_only=True)
async def test_formbot_example(form_bot_agent: Agent):
    def response_for_slot(slot: Text) -> Dict[Text, Any]:
        if slot:
            form = "restaurant_form"
            response = f"utter_ask_{slot}"
        else:
            form = None
            response = "utter_submit"

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
            "responses": [{"response": response}],
        }

    async def mock_form_happy_path(
        input_text: Text, output_text: Text, slot: Optional[Text] = None
    ) -> None:
        with aioresponses() as mocked:
            mocked.post(
                "https://example.com/webhooks/actions",
                payload=response_for_slot(slot),
                repeat=True,
            )
            responses = await form_bot_agent.handle_text(input_text)
            assert responses[0]["text"] == output_text

    async def mock_form_unhappy_path(
        input_text: Text, output_text: Text, slot: Optional[Text]
    ) -> None:
        response_error = {
            "error": f"Failed to extract slot {slot} with action restaurant_form",
            "action_name": "restaurant_form",
        }
        with aioresponses() as mocked:
            # Request which rejects form execution
            mocked.post(
                "https://example.com/webhooks/actions",
                repeat=False,
                exception=ClientResponseError(400, "", json.dumps(response_error)),
            )
            # Request after returning from unhappy path which sets next requested slot
            mocked.post(
                "https://example.com/webhooks/actions",
                payload=response_for_slot(slot),
                repeat=True,
            )
            responses = await form_bot_agent.handle_text(input_text)
            assert responses[0]["text"] == output_text

    await mock_form_happy_path("/request_restaurant", "What cuisine?", slot="cuisine")
    await mock_form_unhappy_path("/chitchat", "chitchat", slot="cuisine")
    await mock_form_happy_path(
        '/inform{"cuisine": "mexican"}', "How many people?", slot="num_people"
    )
    await mock_form_happy_path(
        '/inform{"number": "2"}', "Do you want to sit outside?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "Please provide additional preferences", slot="preferences"
    )

    responses = await form_bot_agent.handle_text("/restart")
    assert responses[0]["text"] == "restarted"

    responses = await form_bot_agent.handle_text("/greet")
    assert (
        responses[0]["text"]
        == "Hello! I am restaurant search assistant! How can I help?"
    )

    await mock_form_happy_path("/request_restaurant", "What cuisine?", slot="cuisine")
    await mock_form_happy_path(
        '/inform{"cuisine": "mexican"}', "How many people?", slot="num_people"
    )
    await mock_form_happy_path(
        '/inform{"number": "2"}', "Do you want to sit outside?", slot="outdoor_seating"
    )
    await mock_form_unhappy_path(
        "/stop", "Do you want to continue?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "Do you want to sit outside?", slot="outdoor_seating"
    )
    await mock_form_happy_path(
        "/affirm", "Please provide additional preferences", slot="preferences"
    )
    await mock_form_happy_path(
        "/deny", "Please give your feedback on your experience so far", slot="feedback"
    )
    await mock_form_happy_path('/inform{"feedback": "great"}', "All done!")

    responses = await form_bot_agent.handle_text("/thankyou")
    assert responses[0]["text"] == "You are welcome :)"
