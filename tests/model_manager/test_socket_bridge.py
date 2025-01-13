from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pytest import CaptureFixture, MonkeyPatch
from socketio import AsyncServer

from rasa.model_manager.runner_service import BotSession
from rasa.model_manager.socket_bridge import socketio_websocket_traffic_wrapper


@pytest.fixture
def deployment_id() -> str:
    return "test_deployment_id"


@pytest.fixture
def running_bots(deployment_id: str) -> Dict[str, Any]:
    running_bot = Mock(spec=BotSession, internal_url="http://test_internal_url")
    running_bot.is_alive.return_value = True
    return {deployment_id: running_bot}


@pytest.fixture
def test_sid() -> str:
    return "test_sid"


@pytest.fixture
def mock_sio() -> AsyncMock:
    return AsyncMock(spec=AsyncServer)


def set_up_mock_create_bridge_client(mock_create_bridge_client: Mock, test_sid: str):
    mock_client = Mock()
    mock_client.sid = test_sid
    mock_create_bridge_client.return_value = mock_client


@patch("rasa.model_manager.socket_bridge.create_bridge_client")
async def test_socketio_websocket_traffic_wrapper_valid_token(
    mock_create_bridge_client: Mock,
    monkeypatch: MonkeyPatch,
    test_public_key: str,
    valid_jwt_token: str,
    deployment_id: str,
    running_bots: Dict[str, Any],
    test_sid: str,
    mock_sio: AsyncMock,
):
    set_up_mock_create_bridge_client(mock_create_bridge_client, test_sid)
    monkeypatch.setattr(
        "rasa.model_manager.studio_jwt_auth.get_public_key_from_keycloak",
        lambda: test_public_key,
    )
    auth = {"deployment_id": deployment_id, "token": valid_jwt_token}

    result = await socketio_websocket_traffic_wrapper(
        mock_sio, running_bots, test_sid, auth
    )

    assert result is True
    mock_create_bridge_client.assert_called_once_with(
        mock_sio, running_bots[deployment_id].internal_url, test_sid, deployment_id
    )


@patch("rasa.model_manager.socket_bridge.create_bridge_client")
async def test_socketio_websocket_traffic_wrapper_invalid_audience(
    mock_create_bridge_client: Mock,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    test_public_key: str,
    jwt_token_with_invalid_audience: str,
    deployment_id: str,
    running_bots: Dict[str, Any],
    test_sid: str,
    mock_sio: AsyncMock,
):
    set_up_mock_create_bridge_client(mock_create_bridge_client, test_sid)
    monkeypatch.setattr(
        "rasa.model_manager.studio_jwt_auth.get_public_key_from_keycloak",
        lambda: test_public_key,
    )
    auth = {"deployment_id": deployment_id, "token": jwt_token_with_invalid_audience}

    result = await socketio_websocket_traffic_wrapper(
        mock_sio, running_bots, test_sid, auth
    )

    assert result is False
    mock_create_bridge_client.assert_not_called()
    captured = capsys.readouterr()
    assert "Invalid JWT token" in captured.out


@patch("rasa.model_manager.socket_bridge.create_bridge_client")
async def test_socketio_websocket_traffic_wrapper_no_token(
    mock_create_bridge_client: Mock,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    test_public_key: str,
    deployment_id: str,
    running_bots: Dict[str, Any],
    test_sid: str,
    mock_sio: AsyncMock,
):
    set_up_mock_create_bridge_client(mock_create_bridge_client, test_sid)
    monkeypatch.setattr(
        "rasa.model_manager.studio_jwt_auth.get_public_key_from_keycloak",
        lambda: test_public_key,
    )
    auth = {"deployment_id": deployment_id}

    result = await socketio_websocket_traffic_wrapper(
        mock_sio, running_bots, test_sid, auth
    )

    assert result is False
    mock_create_bridge_client.assert_not_called()
    captured = capsys.readouterr()
    assert "model_runner.user_no_token" in captured.out
