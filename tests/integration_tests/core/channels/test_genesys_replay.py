"""Pytest tests for the Genesys AudioHook channel using genesys_replay.py.

Exercises the Rasa Genesys connector by replaying minimal AudioHook traffic.
"""

import os
from pathlib import Path

import pytest

from tests.integration_tests.core.channels.utils.genesys_replay import GenesysReplay


@pytest.fixture
def rasa_url() -> str:
    return os.getenv("RASA_URL", "http://localhost:5005")


@pytest.fixture
def traffic_file() -> str:
    default_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "e2e_voice"
        / "genesys_traffic.json"
    )
    return os.getenv("GENESYS_TRAFFIC_FILE", str(default_path))


@pytest.fixture
def websocket_timeout() -> int:
    """Timeout for WebSocket recv (use WEBSOCKET_TIMEOUT to override).

    Default is generous: after the AudioHook ``open`` handshake the server
    connects ASR/TTS and runs the first dialogue turn before further frames;
    cold starts on CI can exceed a few seconds. A short timeout lets the client
    stop waiting and close the socket while the server is still working, which
    can surface as server errors (e.g. 1011).
    """
    return int(os.getenv("WEBSOCKET_TIMEOUT", "30"))


@pytest.fixture
def websocket_delay() -> float:
    return float(os.getenv("WEBSOCKET_DELAY", "0.05"))


@pytest.fixture
def replay_instance(
    rasa_url: str, traffic_file: str, websocket_timeout: int, websocket_delay: float
) -> GenesysReplay:
    if not os.path.exists(traffic_file):
        pytest.skip(f"Traffic file not found: {traffic_file}")

    return GenesysReplay(
        rasa_url=rasa_url,
        log_file=traffic_file,
        timeout=websocket_timeout,
        delay=websocket_delay,
    )


@pytest.mark.asyncio
async def test_genesys_replay_basic_flow(replay_instance: GenesysReplay) -> None:
    """Open is sent and the server responds with opened."""
    await replay_instance.replay_genesys_session()

    assert replay_instance.connection_state[
        "open_message_sent"
    ], "open was not sent or did not reach the server"
    assert replay_instance.connection_state[
        "opened_response_received"
    ], "Did not receive AudioHook opened response from Rasa"


@pytest.mark.asyncio
async def test_genesys_replay_messages_sent(replay_instance: GenesysReplay) -> None:
    await replay_instance.replay_genesys_session()

    n = replay_instance.connection_state["messages_sent"]
    assert n > 0, f"No WebSocket messages sent (got {n})"


@pytest.mark.asyncio
async def test_genesys_replay_responses_received(
    replay_instance: GenesysReplay,
) -> None:
    await replay_instance.replay_genesys_session()

    n = replay_instance.connection_state["messages_received"]
    assert n > 0, f"No WebSocket responses received (got {n})"


@pytest.mark.asyncio
async def test_genesys_replay_no_errors(replay_instance: GenesysReplay) -> None:
    await replay_instance.replay_genesys_session()

    errors = replay_instance.connection_state["errors"]
    bad = [e for e in errors if not e.get("graceful", False)]
    assert not bad, f"Non-graceful errors: {bad}"


@pytest.mark.asyncio
async def test_genesys_replay_critical_flow(replay_instance: GenesysReplay) -> None:
    await replay_instance.replay_genesys_session()

    state = replay_instance.connection_state
    assert state["open_message_sent"], "open step failed"
    assert state["opened_response_received"], "opened step failed"
    assert state["messages_sent"] >= 1
    assert state["messages_received"] >= 1


@pytest.mark.asyncio
async def test_genesys_replay_complete(replay_instance: GenesysReplay) -> None:
    await replay_instance.replay_genesys_session()

    state = replay_instance.connection_state
    assert state["open_message_sent"]
    assert state["opened_response_received"]
    assert state["messages_sent"] > 0
    assert state["messages_received"] > 0
    bad = [e for e in state["errors"] if not e.get("graceful", False)]
    assert not bad, f"Non-graceful errors: {bad}"


@pytest.mark.asyncio
async def test_genesys_replay_connection_state(replay_instance: GenesysReplay) -> None:
    await replay_instance.replay_genesys_session()

    state = replay_instance.connection_state
    assert "connected" in state
    assert "open_message_sent" in state
    assert "opened_response_received" in state
    assert "messages_sent" in state
    assert "messages_received" in state
    assert "errors" in state
    assert isinstance(state["messages_sent"], int)
    assert isinstance(state["messages_received"], int)
    assert isinstance(state["errors"], list)
