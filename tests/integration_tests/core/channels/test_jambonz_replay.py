"""
Pytest tests for Jambonz Stream channel using jambonz_replay.py.

Tests Rasa Jambonz Stream channel by replaying captured traffic.
"""

import os
from pathlib import Path

import pytest

from tests.integration_tests.core.channels.utils.jambonz_replay import (
    JambonzStreamReplay,
)


@pytest.fixture
def rasa_url() -> str:
    """Rasa server URL for testing."""
    return os.getenv("RASA_URL", "http://localhost:5005")


@pytest.fixture
def traffic_file() -> str:
    """Path to captured Jambonz Stream traffic file."""
    default_path = (
        Path(__file__).parent.parent.parent.parent.parent
        / "data"
        / "e2e_voice"
        / "jambonz_traffic.json"
    )
    return os.getenv("JAMBONZ_TRAFFIC_FILE", str(default_path))


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
) -> JambonzStreamReplay:
    """JambonzStreamReplay instance for testing."""
    if not os.path.exists(traffic_file):
        pytest.skip(f"Traffic file not found: {traffic_file}")

    return JambonzStreamReplay(
        rasa_url=rasa_url,
        log_file=traffic_file,
        timeout=websocket_timeout,
        delay=websocket_delay,
    )


@pytest.mark.asyncio
async def test_jambonz_replay_basic_flow(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that the basic Jambonz Stream flow completes successfully."""
    await replay_instance.replay_jambonz_session()

    assert replay_instance.connection_state[
        "metadata_sent"
    ], "Metadata (callSid, from, to) was not sent - first message must be metadata"


@pytest.mark.asyncio
async def test_jambonz_replay_messages_sent(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that messages were successfully sent."""
    await replay_instance.replay_jambonz_session()

    messages_sent: int = replay_instance.connection_state["messages_sent"]
    assert (
        messages_sent > 0
    ), f"No messages were sent (expected > 0, got {messages_sent})"


@pytest.mark.asyncio
async def test_jambonz_replay_responses_received(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that responses were received from Rasa."""
    await replay_instance.replay_jambonz_session()

    messages_received: int = replay_instance.connection_state["messages_received"]
    assert (
        messages_received > 0
    ), f"Invalid messages_received (expected > 0, got {messages_received})"


@pytest.mark.asyncio
async def test_jambonz_replay_no_errors(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that no non-graceful errors occurred."""
    await replay_instance.replay_jambonz_session()

    errors: list = replay_instance.connection_state["errors"]
    actual_errors = [e for e in errors if not e.get("graceful", False)]

    assert (
        len(actual_errors) == 0
    ), f"Encountered {len(actual_errors)} non-graceful errors: {actual_errors}"


@pytest.mark.asyncio
async def test_jambonz_replay_critical_flow(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that the critical flow (metadata sent) completes."""
    await replay_instance.replay_jambonz_session()

    state = replay_instance.connection_state

    assert state["metadata_sent"], "Metadata was not sent"
    assert state["messages_sent"] >= 1, "Expected at least 1 message (metadata)"


@pytest.mark.asyncio
async def test_jambonz_replay_complete(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that verifies the complete replay flow."""
    await replay_instance.replay_jambonz_session()

    state = replay_instance.connection_state

    results = {
        "metadata_sent": state["metadata_sent"],
        "messages_sent": state["messages_sent"],
        "messages_received": state["messages_received"],
        "errors": state["errors"],
    }

    assert results["metadata_sent"], "Metadata was not sent"
    assert (
        results["messages_sent"] > 0
    ), f"Expected messages sent > 0, got {results['messages_sent']}"

    actual_errors: list = [e for e in results["errors"] if not e.get("graceful", False)]
    assert len(actual_errors) == 0, f"Non-graceful errors occurred: {actual_errors}"


@pytest.mark.asyncio
async def test_jambonz_replay_connection_state(
    replay_instance: JambonzStreamReplay,
) -> None:
    """Test that connection state is properly tracked."""
    await replay_instance.replay_jambonz_session()

    state = replay_instance.connection_state

    assert "connected" in state
    assert "metadata_sent" in state
    assert "messages_sent" in state
    assert "messages_received" in state
    assert "errors" in state

    assert isinstance(state["messages_sent"], int)
    assert isinstance(state["messages_received"], int)
    assert isinstance(state["errors"], list)
    assert isinstance(state["metadata_sent"], bool)
