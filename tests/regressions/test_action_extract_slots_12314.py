import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Text

import pytest
from aioresponses import aioresponses

from rasa import train
from rasa.core.agent import load_agent
from rasa.core.channels import UserMessage, CollectingOutputChannel

SENDER = "sender"

BOT_DIRECTORY = (
    Path(os.path.dirname(__file__)).parent.parent
    / "data"
    / "test_action_extract_slots_12314"
)


@pytest.fixture
def model_file():
    if not (BOT_DIRECTORY / "models" / "model.tar.gz").exists():
        output_directory = TemporaryDirectory()
        print(output_directory)

        train(
            domain=str(BOT_DIRECTORY / "domain.yml"),
            config=str(BOT_DIRECTORY / "config.yml"),
            training_files=str(BOT_DIRECTORY / "data"),
            output=str(BOT_DIRECTORY / "models"),
            fixed_model_name="model",
        )

    return str(BOT_DIRECTORY / "models" / "model.tar.gz")


async def test_setting_slot_with_custom_action(model_file: Text):
    """
    Attemps the following conversation:
    User: help me install a rsa token
    Bot: What type of rsa token do you need?
    User: hard
    Bot: User has requested rsa token type of hard
    User: help me install a hard rsa token
    Bot: User has requested rsa token type of hard

    The custom action should set the slot `rsa_token` to
    the value `hard` in the second and third message.
    A `SlotSet` event should be emitted in the third message.

    From the Jira Comment by Chris,
    https://rasahq.atlassian.net/browse/ATO-678?focusedCommentId=18017
    """
    agent = await load_agent(model_path=model_file)
    output_channel = CollectingOutputChannel()

    # Check that the bot asks for the type of RSA token
    with aioresponses() as mocked:
        mocked.post(
            "http://localhost:5055/webhook",
            payload={
                "events": [
                    {
                        "event": "slot",
                        "name": "rsa_token",
                        "value": "unknown",
                    }
                ]
            },
        )

        # Send the first message
        await agent.handle_message(
            _build_user_message(output_channel, "help me install a rsa token")
        )

    assert output_channel.messages[-1] == {
        "recipient_id": SENDER,
        "text": "What type of rsa token do you need?",
    }

    # Send the second message
    await agent.handle_message(_build_user_message(output_channel, "hard"))

    # Check that the bot confirms that the user has
    # requested for an RSA token of type hard
    assert output_channel.messages[-1] == {
        "recipient_id": SENDER,
        "text": "User has requested rsa token type of hard",
    }

    # Send the third message
    with aioresponses() as mocked:
        mocked.post(
            "http://localhost:5055/webhook",
            payload={
                "events": [
                    {
                        "event": "slot",
                        "name": "rsa_token",
                        "value": "hard",
                    }
                ]
            },
        )

        await agent.handle_message(
            _build_user_message(output_channel, "help me install a hard rsa token")
        )

    # Check that the bot confirms that the user
    # has requested for an RSA token of type hard
    assert output_channel.messages[-1] == {
        "recipient_id": SENDER,
        "text": "User has requested rsa token type of hard",
    }

    # Check that the SlotSet event was emitted
    events = await agent.latest_message.parse_data()["events"]
    assert any(
        event["event"] == "slot" and event["name"] == "rsa_token" for event in events
    )


def _build_user_message(output_channel, text):
    return UserMessage(text=text, sender_id=SENDER, output_channel=output_channel)
