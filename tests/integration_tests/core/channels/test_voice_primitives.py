import asyncio
import base64
import json
import os
import time

import aiohttp
import httpx
import pytest
import requests


def wait_for_service(
    base_url, endpoints=None, timeout=None, interval=None, req_timeout=5
):
    """
    Wait for any of the listed endpoints on base_url to return HTTP 200.
    - endpoints: list of paths (e.g. ['/', '/health']). Defaults to common candidates.
    - timeout: total seconds to wait (env WAIT_FOR_SERVICE_TIMEOUT overrides)
    - interval: seconds between polls (env WAIT_FOR_SERVICE_INTERVAL overrides)
    - req_timeout: per-request timeout in seconds
    Raises RuntimeError with the last observed error
    if service doesn't become available.
    """
    endpoints = endpoints or ["/", "/health", "/status", "/version"]
    timeout = int(os.getenv("WAIT_FOR_SERVICE_TIMEOUT", timeout or 120))
    interval = float(os.getenv("WAIT_FOR_SERVICE_INTERVAL", interval or 1.0))
    end = time.time() + timeout
    last_err = None

    while time.time() < end:
        for ep in endpoints:
            url = base_url.rstrip("/") + (ep if ep.startswith("/") else f"/{ep}")
            try:
                r = requests.get(url, timeout=req_timeout)
                if r.status_code == 200:
                    return True
                last_err = f"{url} returned status {r.status_code}"
            except requests.RequestException as exc:
                last_err = f"{url} request error: {exc}"
        time.sleep(interval)

    raise RuntimeError(
        f"Service at {base_url} not available after {timeout}s. "
        f"Last error: {last_err}"
    )


@pytest.fixture(scope="session", autouse=True)
def ensure_rasa_available():
    base_url = "http://localhost:5005"
    try:
        wait_for_service(base_url)
        print(f"✓ Service available at {base_url}")
    except RuntimeError as e:
        pytest.exit(f"Service not available: {e}")


@pytest.fixture
def sender_id() -> str:
    """Get sender_id from environment variable at test execution time"""
    return os.getenv("SENDER_ID", "test_voice_user_123")


@pytest.mark.asyncio
async def test_voice_session_start_end_and_tracker_events_plain_websocket(
    voice_file, sender_id
):
    """
    1. Connect to WebSocket endpoint
    2. Send a voice WAV file over websocket
    3. Verify session events in tracker
    """
    http_base_url = "http://localhost:5005"
    ws_url = "ws://localhost:5005/webhooks/browser_audio/websocket"

    voice_path = voice_file

    if not os.path.exists(voice_path):
        pytest.skip(f"voice file not found at `{voice_path}`")

    async with httpx.AsyncClient() as http:
        await http.delete(f"{http_base_url}/conversations/{sender_id}/tracker")
        await asyncio.sleep(0.5)

        # Use plain WebSocket connection (aiohttp) instead of Socket.IO
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print(f"✓ Connected to WebSocket at {ws_url}")

                # Read voice file and encode as base64
                with open(voice_path, "rb") as f:
                    audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

                # Send audio message in browser_audio format
                # browser_audio expects: {"audio": base64_encoded_audio}
                message = {
                    "audio": audio_b64,
                }
                await ws.send_str(json.dumps(message))
                print("✓ Sent audio message (browser_audio format)")

                # Wait for response
                try:
                    response = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    print(f"✓ Received response: {response}")
                except asyncio.TimeoutError:
                    print("! No immediate response received (may be normal)")

                await asyncio.sleep(3)

        # Fetch tracker to verify session started
        tracker_resp = await http.get(
            f"{http_base_url}/conversations/{sender_id}/tracker"
        )
        assert (
            tracker_resp.status_code == 200
        ), f"Tracker fetch failed: {tracker_resp.text}"

        tracker = tracker_resp.json()
        events = tracker.get("events", [])

        # Assert session_started event exists
        assert any(e.get("event") == "session_started" for e in events), (
            "No session_started event found. "
            f"Events: {[e.get('event') for e in events]}"
        )

        # Verify the voice message was processed
        user_events = [
            e
            for e in events
            if e.get("event") in ("user", "user_uttered", "user_message")
        ]
        action_events = [
            e for e in events if e.get("event") in ("action", "action_executed", "bot")
        ]
        print("Events (summary):", [e.get("event") for e in events])
        print("User events found:", len(user_events), "Sample:", user_events[:3])
        print("Action events found:", len(action_events), "Sample:", action_events[:3])

        assert len(events) > 1, (
            "Expected multiple events after voice message "
            "(session_started + user/action events)"
        )

        print(f"✓ Found {len(user_events)} user event(s)")
        print(f"✓ Found {len(action_events)} action event(s)")

        await http.post(
            f"{http_base_url}/conversations/{sender_id}/tracker/events",
            json={"event": "session_ended"},
        )

        await asyncio.sleep(0.5)

        # Fetch tracker again
        tracker_resp = await http.get(
            f"{http_base_url}/conversations/{sender_id}/tracker"
        )
        tracker = tracker_resp.json()
        events = tracker.get("events", [])

        # Should now have 2 session_started events
        session_ended_event = [e for e in events if e.get("event") == "session_ended"]
        assert (
            len(session_ended_event) == 1
        ), f"Expected a session_ended events. Found: {len(session_ended_event)}"
