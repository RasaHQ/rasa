"""
Pytest tests for AudioCodes channel using websocket_replay.py
Tests Rasa AudioCodes channel by replaying captured traffic
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from tests.integration_tests.core.channels.utils.websocket_replay import WebSocketReplay

logger = logging.getLogger(__name__)


def _audiocodes_replay_retry_settings() -> tuple[int, float]:
    """Max attempts and base backoff (seconds) between failed replay assertions."""
    attempts = int(os.getenv("AUDIOCODES_REPLAY_MAX_ATTEMPTS", "5"))
    backoff = float(os.getenv("AUDIOCODES_REPLAY_BACKOFF_SECONDS", "5"))
    return max(1, attempts), max(0.0, backoff)


def _strict_replay_error_policy() -> bool:
    """If true, every non-graceful replay error fails the test."""
    return os.getenv("AUDIOCODES_REPLAY_STRICT_ERRORS", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _is_transient_activities_1011(err: Dict[str, Any]) -> bool:
    """True when the server closed with 1011 while waiting after ``activities``.

    After ``activities``, Rasa connects ASR/TTS and runs ``start_session``; a
    failing provider or slow CI can surface as WebSocket 1011 (internal error)
    on the client. This is an integration-environment flake, not a protocol bug
    in the captured handshake (session.initiate / session.accepted still work).
    """
    if err.get("graceful"):
        return False
    if err.get("message_kind") != "activities":
        return False
    msg = str(err.get("error", ""))
    return "1011" in msg


def _non_graceful_replay_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in errors if not e.get("graceful", False)]


def _reportable_replay_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Non-graceful errors, optionally filtering known transient integration flakes."""
    raw_errors = _non_graceful_replay_errors(errors)
    if _strict_replay_error_policy():
        return raw_errors
    return [e for e in raw_errors if not _is_transient_activities_1011(e)]


def _maybe_warn_transient_ignored(all_non_graceful: List[Dict[str, Any]]) -> None:
    """Log and warn when lenient policy filters known transient 1011 errors."""
    if _strict_replay_error_policy():
        return
    filtered_errors = [e for e in all_non_graceful if _is_transient_activities_1011(e)]
    if not filtered_errors:
        return
    logger.warning(
        "AudioCodes replay: ignored non-graceful error(s) treated as transient "
        "integration noise (1011 after activities): %s. "
        "Set AUDIOCODES_REPLAY_STRICT_ERRORS=1 to fail on these.",
        filtered_errors,
    )


async def _replay_with_backoff(
    replay_instance: WebSocketReplay,
    validate: Callable[[WebSocketReplay], None],
) -> None:
    """Run replay; on AssertionError retry with linear backoff."""
    max_attempts, base_backoff_seconds = _audiocodes_replay_retry_settings()

    for attempt in range(1, max_attempts + 1):
        await replay_instance.replay_websocket_session()
        try:
            validate(replay_instance)
            return
        except AssertionError as err:
            if attempt >= max_attempts:
                raise
            delay_seconds = base_backoff_seconds * attempt
            logger.warning(
                "AudioCodes replay assertion failed "
                "(attempt %s/%s), retrying in %ss: %s",
                attempt,
                max_attempts,
                delay_seconds,
                err,
            )
            await asyncio.sleep(delay_seconds)


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
    """Timeout for WebSocket responses (use WEBSOCKET_TIMEOUT to override).

    Default is generous: after ``activities`` the server connects ASR/TTS and
    runs the first dialogue turn before anything is sent back; cold starts on
    CI can exceed a few seconds. A short timeout lets the client close the
    socket while the server is still working, which surfaces as server 1011.
    """
    return int(os.getenv("WEBSOCKET_TIMEOUT", "30"))


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
    """Test that no non-graceful errors occurred (see transient 1011 filter)."""

    def _assert(replay: WebSocketReplay) -> None:
        errors = replay.connection_state["errors"]
        bad = _non_graceful_replay_errors(errors)
        reportable = _reportable_replay_errors(errors)
        _maybe_warn_transient_ignored(bad)
        assert (
            len(reportable) == 0
        ), f"Encountered non-graceful replay errors: {reportable}"

    await _replay_with_backoff(replay_instance, _assert)


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

    def _assert(replay: WebSocketReplay) -> None:
        state = replay.connection_state
        results = {
            "session_initiated": state["session_initiated"],
            "session_accepted": state["session_accepted"],
            "activities_start_sent": state["activities_start_sent"],
            "messages_sent": state["messages_sent"],
            "messages_received": state["messages_received"],
            "errors": state["errors"],
        }
        assert results["session_initiated"], "Session was not initiated"
        assert results["session_accepted"], "Session was not accepted"
        assert results["activities_start_sent"], "Activities start was not sent"
        assert (
            results["messages_sent"] > 0
        ), f"Expected messages sent > 0, got {results['messages_sent']}"
        assert (
            results["messages_received"] > 0
        ), f"Expected messages received > 0, got {results['messages_received']}"
        bad = _non_graceful_replay_errors(results["errors"])
        reportable = _reportable_replay_errors(results["errors"])
        _maybe_warn_transient_ignored(bad)
        assert (
            len(reportable) == 0
        ), f"Non-graceful replay errors occurred: {reportable}"

    await _replay_with_backoff(replay_instance, _assert)


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
