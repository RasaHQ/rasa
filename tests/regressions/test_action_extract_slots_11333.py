import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Text

import pytest

from rasa import train
from rasa.core.agent import load_agent
from rasa.core.channels import UserMessage, CollectingOutputChannel

SENDER = "sender"

BOT_DIRECTORY = (
    Path(os.path.dirname(__file__)).parent.parent
    / "data"
    / "test_action_extract_slots_11333"
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


@pytest.mark.flaky
async def test_retaining_slot_values_with_augmented_memoization(model_file: Text):
    agent = await load_agent(model_path=model_file)

    output_channel = CollectingOutputChannel()

    await agent.handle_message(
        _build_user_message(output_channel, "Block my savings account")
    )
    await agent.handle_message(_build_user_message(output_channel, "Hi"))
    await agent.handle_message(_build_user_message(output_channel, "Hi"))
    await agent.handle_message(
        _build_user_message(output_channel, "Block my savings account")
    )

    assert output_channel.messages[-1] == {
        "recipient_id": SENDER,
        "text": "your account has been blocked",
    }


def _build_user_message(output_channel, text):
    return UserMessage(text=text, sender_id=SENDER, output_channel=output_channel)
