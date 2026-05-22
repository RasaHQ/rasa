import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rasa.core.channels.voice_stream.audio_bytes import (
    MULAW_8KHZ,
    AudioFormat,
)
from rasa.core.channels.voice_stream.call_state import CallState, _call_state


@pytest.fixture
def mock_validate_voice_license_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock the validate_voice_license_scope function."""
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.voice_channel.validate_voice_license_scope",
        MagicMock(),
    )


@pytest.fixture
def setup_call_state():
    """Setup and teardown call state for voice-stream tests.

    This fixture centralizes call state initialization so all voice-stream
    channel tests can rely on a bound context for `_call_state`.
    """
    # Initialize a new call state
    _call_state.set(
        CallState(
            internal_queue=asyncio.Queue(),
        )
    )
    yield
    # Reset call state to a fresh CallState instance to avoid unbound errors
    try:
        _call_state.set(
            CallState(
                internal_queue=asyncio.Queue(),
            )
        )
    except Exception:
        # Best-effort cleanup; if this fails, ignore to not mask test failures
        pass


@pytest.fixture
def mulaw_format() -> AudioFormat:
    return MULAW_8KHZ


@pytest.fixture
def azure_sdk_mocks(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Patch Azure Speech SDK; return mock stream and recognizer for connect tests."""
    mock_stream = MagicMock()
    mock_recognizer = MagicMock()
    mock_recognizer.start_continuous_recognition_async = MagicMock(return_value=None)
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.SpeechConfig",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.audio.AudioStreamFormat",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.audio.PushAudioInputStream",
        MagicMock(return_value=mock_stream),
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.audio.AudioConfig",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.SpeechRecognizer",
        MagicMock(return_value=mock_recognizer),
    )
    return SimpleNamespace(stream=mock_stream, recognizer=mock_recognizer)


async def wait_for_task_to_become_cancelled(
    task: asyncio.Task,
) -> None:
    try:
        await task
    except asyncio.CancelledError:
        assert task.cancelled()
