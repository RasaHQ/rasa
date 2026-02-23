import warnings
from asyncio import AbstractEventLoop
from pathlib import Path
from time import sleep, time
from typing import Text
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from sanic import Sanic

import rasa.shared.core.domain
from rasa.core import run
from rasa.core.available_agents import AvailableAgents
from rasa.core.brokers.sql import SQLEventBroker
from rasa.core.config.available_endpoints import AvailableEndpoints
from rasa.core.config.credentials import CredentialsConfig
from rasa.core.run import serve_application

CREDENTIALS_FILE = "data/test_moodbot/credentials.yml"


def test_create_http_input_channels():
    credentials_config = CredentialsConfig.load_from_file(Path(CREDENTIALS_FILE))
    channels = run.create_input_channels(None, credentials_config)
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
    credentials_config = CredentialsConfig.load_from_file(Path(CREDENTIALS_FILE))

    channels = run.create_input_channels("facebook", credentials_config)
    assert len(channels) == 1
    assert channels[0].name() == "facebook"


def test_create_single_input_channels_by_class():
    credentials_config = CredentialsConfig.load_from_file(Path(CREDENTIALS_FILE))
    channels = run.create_input_channels(
        "rasa.core.channels.rest.RestInput", credentials_config
    )
    assert len(channels) == 1
    assert channels[0].name() == "rest"


def test_create_single_input_channels_by_class_wo_credentials():
    channels = run.create_input_channels(
        "rasa.core.channels.rest.RestInput", credentials_config=None
    )

    assert len(channels) == 1
    assert channels[0].name() == "rest"


async def test_load_agent_on_start_with_good_model_file(
    trained_rasa_model: Text, rasa_server: Sanic, loop: AbstractEventLoop
):
    agent = await run.load_agent_on_start(
        trained_rasa_model,
        AvailableEndpoints(),
        None,
        AvailableAgents(),
        rasa_server,
        loop,
    )

    start_time = time()
    delay_in_sec = 2

    # Poll for upto 30 sec (with increasing delay) for agent to be ready
    while not agent.is_ready():
        sleep(delay_in_sec)
        delay_in_sec += delay_in_sec + 2
        if time() - start_time > 30:
            break

    assert agent.is_ready()
    assert isinstance(agent.domain, rasa.shared.core.domain.Domain)


async def test_load_agent_on_start_with_bad_model_file(
    tmp_path: Path, rasa_non_trained_server: Sanic, loop: AbstractEventLoop
):
    fake_model = tmp_path / "fake_model.tar.gz"
    fake_model.touch()
    fake_model_path = str(fake_model)

    with pytest.warns(UserWarning) as warnings:
        await run.load_agent_on_start(
            fake_model_path,
            AvailableEndpoints(),
            None,
            AvailableAgents(),
            rasa_non_trained_server,
            loop,
        )
        assert any("No valid model found at" in str(w.message) for w in warnings)


async def test_close_resources(loop: AbstractEventLoop):
    broker = SQLEventBroker()
    app = Mock()
    app.ctx.agent = Mock()
    app.ctx.agent.close = AsyncMock()
    app.ctx.agent.tracker_store.event_broker = broker
    app.ctx.agent.privacy_manager = None

    with warnings.catch_warnings() as record:
        await run.close_resources(app, loop)
        assert record is None

    app.ctx.agent.close.assert_called_once()


@pytest.mark.parametrize("inspect", [False, True])
def test_is_inspector_enabled_param_initialisation(
    inspect: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that the inspector is enabled or disabled based on the parameter inspect"""
    mock_app = MagicMock(spec=Sanic)
    mock_configure_app = MagicMock(return_value=mock_app)
    monkeypatch.setattr("rasa.core.run.configure_app", mock_configure_app)
    mock_create_http_input_channels = MagicMock()
    monkeypatch.setattr(
        "rasa.core.run.create_input_channels", mock_create_http_input_channels
    )
    mock_telemetry_track_server_start = MagicMock()
    monkeypatch.setattr(
        "rasa.core.run.telemetry.track_server_start", mock_telemetry_track_server_start
    )
    serve_application(inspect=inspect)

    configure_app_call_args = mock_configure_app.call_args
    assert configure_app_call_args.kwargs.get("is_inspector_enabled") == inspect
    mock_app.run.assert_called_once()
