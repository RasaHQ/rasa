"""
Pytest tests for AudioCodes channel using websocket_replay.py
Tests Rasa AudioCodes channel by replaying captured traffic
"""

import os
from pathlib import Path

import pytest

from tests.integration_tests.core.channels.utils.websocket_replay import WebSocketReplay


@pytest.fixture
def rasa_url() -> str:
    """Rasa server URL for testing"""
    return os.getenv("RASA_URL", "http://localhost:5005")


@pytest.fixture
def traffic_file() -> str:
    """Path to captured AudioCodes traffic file"""
    # Default to audiocodes_traffic.json in data/e2e_voice/
    # From tests/integration_tests/core/channels/ ->
    # workspace root is 5 levels up
    default_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "e2e_voice"
        / "audiocodes_traffic.json"
    )
    return os.getenv("AUDIOCODES_TRAFFIC_FILE", str(default_path))


@pytest.fixture
def websocket_timeout() -> int:
    """Timeout for WebSocket responses (use WEBSOCKET_TIMEOUT to override)."""
    return int(os.getenv("WEBSOCKET_TIMEOUT", "3"))


@pytest.fixture
def websocket_delay() -> float:
    """Delay between messages (increase via WEBSOCKET_DELAY for debugging)."""
    return float(os.getenv("WEBSOCKET_DELAY", "0.05"))


@pytest.fixture
def replay_instance(
    rasa_url: str, traffic_file: str, websocket_timeout: int, websocket_delay: float
) -> WebSocketReplay:
    """Create a WebSocketReplay instance for testing"""
    if not os.path.exists(traffic_file):
        pytest.skip(f"Traffic file not found: {traffic_file}")

    return WebSocketReplay(
        rasa_url=rasa_url,
        log_file=traffic_file,
        timeout=websocket_timeout,
        delay=websocket_delay,
    )


@pytest.mark.asyncio
async def test_audiocodes_replay_basic_flow(replay_instance: WebSocketReplay):
    """Test that the basic AudioCodes flow completes successfully"""
    # Run the replay
    await replay_instance.replay_websocket_session()

    # Assert session was initiated
    assert replay_instance.connection_state["session_initiated"], (
        "Session was not initiated - session.initiate message was not "
        "sent or processed"
    )

    # Assert session was accepted
    assert replay_instance.connection_state[
        "session_accepted"
    ], "Session was not accepted by Rasa - did not receive session.accepted response"

    # Assert activities start message was sent
    assert replay_instance.connection_state["activities_start_sent"], (
        "Activities start message was not sent - required for call "
        "parameters collection"
    )


@pytest.mark.asyncio
async def test_audiocodes_replay_messages_sent(replay_instance: WebSocketReplay):
    """Test that messages were successfully sent"""
    await replay_instance.replay_websocket_session()

    messages_sent = replay_instance.connection_state["messages_sent"]
    assert (
        messages_sent > 0
    ), f"No messages were sent (expected > 0, got {messages_sent})"


@pytest.mark.asyncio
async def test_audiocodes_replay_responses_received(replay_instance: WebSocketReplay):
    """Test that responses were received from Rasa"""
    await replay_instance.replay_websocket_session()

    messages_received = replay_instance.connection_state["messages_received"]
    assert (
        messages_received > 0
    ), f"No responses were received from Rasa (expected > 0, got {messages_received})"


@pytest.mark.asyncio
async def test_audiocodes_replay_no_errors(replay_instance: WebSocketReplay):
    """Test that no non-graceful errors occurred"""
    await replay_instance.replay_websocket_session()

    errors = replay_instance.connection_state["errors"]
    actual_errors = [e for e in errors if not e.get("graceful", False)]

    assert (
        len(actual_errors) == 0
    ), f"Encountered {len(actual_errors)} non-graceful errors: {actual_errors}"


@pytest.mark.asyncio
async def test_audiocodes_replay_critical_flow(replay_instance: WebSocketReplay):
    """Test that the critical flow (initiate -> accepted -> activities start)
    completes"""
    await replay_instance.replay_websocket_session()

    state = replay_instance.connection_state

    assert state["session_initiated"], "Session initiate step failed"
    assert state["session_accepted"], "Session accepted step failed"
    assert state["activities_start_sent"], "Activities start step failed"

    # Verify the flow happened in the right order by checking message counts
    assert (
        state["messages_sent"] >= 2
    ), "Expected at least 2 messages (session.initiate + activities start)"
    assert (
        state["messages_received"] >= 1
    ), "Expected at least 1 response (session.accepted)"


@pytest.mark.asyncio
async def test_audiocodes_replay_complete(replay_instance: WebSocketReplay):
    """Comprehensive test that verifies the complete replay flow"""
    # Run the replay
    await replay_instance.replay_websocket_session()

    state = replay_instance.connection_state

    # Collect all assertions
    results = {
        "session_initiated": state["session_initiated"],
        "session_accepted": state["session_accepted"],
        "activities_start_sent": state["activities_start_sent"],
        "messages_sent": state["messages_sent"],
        "messages_received": state["messages_received"],
        "errors": state["errors"],
    }

    # Assert critical flow
    assert results["session_initiated"], "Session was not initiated"
    assert results["session_accepted"], "Session was not accepted"
    assert results["activities_start_sent"], "Activities start was not sent"

    # Assert message exchange
    assert (
        results["messages_sent"] > 0
    ), f"Expected messages sent > 0, got {results['messages_sent']}"
    assert results["messages_received"] > 0, (
        f"Expected messages received > 0, " f"got {results['messages_received']}"
    )

    # Assert no non-graceful errors
    actual_errors = [e for e in results["errors"] if not e.get("graceful", False)]
    assert len(actual_errors) == 0, f"Non-graceful errors occurred: {actual_errors}"

    return results


@pytest.mark.asyncio
async def test_audiocodes_replay_connection_state(replay_instance: WebSocketReplay):
    """Test that connection state is properly tracked"""
    await replay_instance.replay_websocket_session()

    state = replay_instance.connection_state

    # Verify connection state structure
    assert "connected" in state
    assert "session_initiated" in state
    assert "session_accepted" in state
    assert "activities_start_sent" in state
    assert "messages_sent" in state
    assert "messages_received" in state
    assert "errors" in state

    # Verify state types
    assert isinstance(state["messages_sent"], int)
    assert isinstance(state["messages_received"], int)
    assert isinstance(state["errors"], list)
    assert isinstance(state["session_initiated"], bool)
    assert isinstance(state["session_accepted"], bool)
    assert isinstance(state["activities_start_sent"], bool)
