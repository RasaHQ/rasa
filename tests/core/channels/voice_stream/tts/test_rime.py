import base64
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, L16_48KHZ, MULAW_8KHZ
from rasa.core.channels.voice_stream.tts.rime import RimeTTS, RimeTTSConfig
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation(mulaw_format):
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            RimeTTS(rasa_language="en", format=mulaw_format)
        assert e.match(RimeTTS.required_env_vars[0])
        assert e.match("TTS Engine RimeTTS")


def test_default_config():
    config = RimeTTS.get_default_config("en")
    assert "en" in config.language_map
    assert config.language_map["en"].voice == "cove"
    assert config.language_map["en"].language == "eng"
    assert config.model_id == "mistv2"
    assert config.timeout == 30
    assert config.endpoint == "wss://users.rime.ai/ws2"
    assert config.speed_alpha == 1.0
    assert config.segment == "immediate"
    assert config.no_text_normalization is False


async def test_tts_session_sharing(monkeypatch: MonkeyPatch, mulaw_format):
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)
    tts_engine_2 = RimeTTS(rasa_language="en", format=mulaw_format)
    assert tts_engine_2.session is tts_engine.session


async def test_websocket_url_creation(monkeypatch: MonkeyPatch, mulaw_format):
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)
    ws_url = tts_engine.get_websocket_url()
    assert "wss://users.rime.ai/ws2" in ws_url
    assert "speaker=cove" in ws_url
    assert "modelId=mistv2" in ws_url
    assert "lang=eng" in ws_url
    assert "audioFormat=mulaw" in ws_url
    assert f"samplingRate={mulaw_format.sample_rate}" in ws_url


async def test_websocket_url_with_optional_params(
    monkeypatch: MonkeyPatch, mulaw_format
):
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    config = RimeTTSConfig(
        model_id="mistv2",
        speed_alpha=1.5,
        segment="sentence",
        language_map={
            "en": {"language": "eng", "voice": "aria"},
        },
    )
    tts_engine = RimeTTS(rasa_language="en", config=config, format=mulaw_format)
    ws_url = tts_engine.get_websocket_url()
    assert "speedAlpha=1.5" in ws_url
    assert "segment=sentence" in ws_url


def test_request_headers(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RIME_API_KEY", "test_api_key")
    headers = RimeTTS.get_request_headers()
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_api_key"


async def test_signal_text_done_resets_context_id(
    monkeypatch: MonkeyPatch, mulaw_format
):
    """Test that context_id gets reset every time signal_text_done is called."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # Mock the websocket
    mock_ws = AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    # Get the initial context_id
    initial_context_id = tts_engine.context_id

    # Call signal_text_done
    await tts_engine.signal_text_done()

    # Verify that context_id has been reset to a new value
    assert tts_engine.context_id != initial_context_id

    # Verify the flush operation was sent with the correct format
    mock_ws.send_json.assert_called_once_with({"operation": "flush"})

    # Call signal_text_done again and verify context_id changes again
    second_context_id = tts_engine.context_id
    await tts_engine.signal_text_done()
    assert tts_engine.context_id != second_context_id
    assert tts_engine.context_id != initial_context_id


async def test_signal_text_done_raises_error_when_ws_not_connected(
    monkeypatch: MonkeyPatch, mulaw_format
):
    """Test that signal_text_done raises TTSError when WebSocket is not connected."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # WebSocket is None (not connected)
    tts_engine.ws = None

    with pytest.raises(TTSError, match="WebSocket connection not established"):
        await tts_engine.signal_text_done()


async def test_signal_text_done_raises_error_when_ws_closed(
    monkeypatch: MonkeyPatch, mulaw_format
):
    """Test that signal_text_done raises TTSError when WebSocket is closed."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # Mock a closed websocket
    mock_ws = AsyncMock()
    mock_ws.closed = True
    tts_engine.ws = mock_ws

    with pytest.raises(TTSError, match="WebSocket connection not established"):
        await tts_engine.signal_text_done()


async def test_send_text_chunk_includes_context_id(
    monkeypatch: MonkeyPatch, mulaw_format
):
    """Test that send_text_chunk sends text with the current context_id."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # Mock the websocket
    mock_ws = AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    test_text = "Hello, world!"
    current_context_id = tts_engine.context_id

    await tts_engine.send_text_chunk(test_text)

    mock_ws.send_json.assert_called_once_with(
        {
            "text": test_text,
            "contextId": current_context_id,
        }
    )


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        # L16_48KHZ, Rime doesn't support 48KHz
    ],
)
async def test_configuration_format(format, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS.from_config_dict(config={}, rasa_language="en", format=format)
    assert tts_engine.audio_format == format


@pytest.mark.parametrize(
    "format, expected_encoding",
    [
        (MULAW_8KHZ, "mulaw"),
        (L16_24KHZ, "pcm"),
    ],
)
async def test_get_websocket_url(format, expected_encoding, monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS.from_config_dict(config={}, rasa_language="en", format=format)
    ws_url = tts_engine.get_websocket_url()

    assert f"audioFormat={expected_encoding}" in ws_url
    assert f"samplingRate={format.sample_rate}" in ws_url


async def test_unsupported_format_raises_error(monkeypatch: MonkeyPatch):
    """Test that using an unsupported format (48kHz) raises a TTSError."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")

    with pytest.raises(TTSError, match="Unsupported audio format"):
        tts_engine = RimeTTS.from_config_dict(
            config={}, rasa_language="en", format=L16_48KHZ
        )
        tts_engine.get_websocket_url()


async def test_signal_interrupt_sends_clear(monkeypatch: MonkeyPatch, mulaw_format):
    """Test that signal_interrupt sends a clear payload to Rime."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS.from_config_dict(
        config={}, rasa_language="en", format=mulaw_format
    )

    mock_ws = mock.AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    await tts_engine.signal_interrupt()

    clear_calls = [
        call
        for call in mock_ws.send_json.call_args_list
        if call[0][0].get("operation") == "clear"
    ]
    assert len(clear_calls) == 1


async def test_stream_audio_stops_on_done(monkeypatch: MonkeyPatch, mulaw_format):
    """Test that stream_audio stops yielding when it receives a done message."""
    monkeypatch.setenv("RIME_API_KEY", "test_key")
    tts_engine = RimeTTS.from_config_dict(
        config={}, rasa_language="en", format=mulaw_format
    )

    audio_data = base64.b64encode(b"\x00\x01\x02\x03").decode()
    # ws should receive 2 chunks and stop after done message is received
    ws_messages = [
        {"type": "chunk", "data": audio_data},
        {"type": "chunk", "data": audio_data},
        {"type": "done"},
        {"type": "chunk", "data": audio_data},
    ]

    class MockWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self.send_json = mock.AsyncMock()

        def __aiter__(self):
            return self._generate()

        async def _generate(self):
            for msg_data in ws_messages:
                msg = mock.MagicMock()
                msg.json.return_value = msg_data
                yield msg

    tts_engine.ws = MockWebSocket()

    received_chunks = [chunk async for chunk in tts_engine.stream_audio()]

    assert len(received_chunks) == 2
