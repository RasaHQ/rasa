import pytest
from typing import Text

import rasa.shared.core.domain
import rasa.shared.nlu.interpreter
from sanic import Sanic
from asyncio import AbstractEventLoop
from pathlib import Path
from rasa.core import run, interpreter, policies
from rasa.core.utils import AvailableEndpoints

CREDENTIALS_FILE = "examples/moodbot/credentials.yml"


def test_create_http_input_channels():
    channels = run.create_http_input_channels(None, CREDENTIALS_FILE)
    assert len(channels) == 7

    # ensure correct order
    assert {c.name() for c in channels} == {
        "twilio",
        "slack",
        "telegram",
        "mattermost",
        "facebook",
        "webexteams",
        "rocketchat",
    }


def test_create_single_input_channels():
    channels = run.create_http_input_channels("facebook", CREDENTIALS_FILE)
    assert len(channels) == 1
    assert channels[0].name() == "facebook"


def test_create_single_input_channels_by_class():
    channels = run.create_http_input_channels(
        "rasa.core.channels.rest.RestInput", CREDENTIALS_FILE
    )
    assert len(channels) == 1
    assert channels[0].name() == "rest"


def test_create_single_input_channels_by_class_wo_credentials():
    channels = run.create_http_input_channels(
        "rasa.core.channels.rest.RestInput", credentials_file=None
    )

    assert len(channels) == 1
    assert channels[0].name() == "rest"


async def test_load_agent_on_start_with_good_model_file(
    trained_rasa_model: Text, rasa_server: Sanic, loop: AbstractEventLoop
):
    agent = await run.load_agent_on_start(
        trained_rasa_model, AvailableEndpoints(), None, rasa_server, loop
    )

    assert isinstance(agent.interpreter, interpreter.RasaNLUInterpreter)
    assert isinstance(agent.policy_ensemble, policies.PolicyEnsemble)
    assert isinstance(agent.domain, rasa.shared.core.domain.Domain)


async def test_load_agent_on_start_with_bad_model_file(
    tmp_path: Path, rasa_server: Sanic, loop: AbstractEventLoop
):
    fake_model = tmp_path / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)

    with pytest.warns(UserWarning) as warnings:
        agent = await run.load_agent_on_start(
            fake_model_path, AvailableEndpoints(), None, rasa_server, loop
        )
        assert any(
            "fake_model.tar.gz' could not be loaded" in str(w.message) for w in warnings
        )

    # Fallback agent was loaded even if model was unusable
    assert isinstance(agent.interpreter, rasa.shared.nlu.interpreter.RegexInterpreter)
    assert agent.policy_ensemble is None
    assert isinstance(agent.domain, rasa.shared.core.domain.Domain)
