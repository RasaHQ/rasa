from unittest import mock

import pytest
from aiohttp import WSMsgType
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, L16_48KHZ, MULAW_8KHZ
from rasa.core.channels.voice_stream.tts.deepgram import DeepgramTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation(mulaw_format):
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramTTS(rasa_language="en", format=mulaw_format)
        assert e.match(DeepgramTTS.required_env_vars[0])
        assert e.match("TTS Engine DeepgramTTS")


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch, mulaw_format):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "bad_key")
    tts_engine = DeepgramTTS(rasa_language="en", format=mulaw_format)
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = DeepgramTTS.get_default_config("en")
    assert config.endpoint == "wss://api.deepgram.com/v1/speak"
    assert "en" in config.language_map
    assert config.language_map["en"].model == "aura-2-andromeda-en"


async def test_tts_session_sharing(mulaw_format):
    tts_engine = DeepgramTTS(rasa_language="en", format=mulaw_format)
    tts_engine_2 = DeepgramTTS(rasa_language="en", format=mulaw_format)
    assert tts_engine_2.session is tts_engine.session


@pytest.mark.parametrize(
    "format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_configuration_format(format):
    tts_engine = DeepgramTTS.from_config_dict(
        config={}, rasa_language="en", format=format
    )
    assert tts_engine.audio_format == format


@pytest.mark.parametrize(
    "format, expected_encoding",
    [
        (MULAW_8KHZ, "mulaw"),
        (L16_24KHZ, "linear16"),
        (L16_48KHZ, "linear16"),
    ],
)
async def test_get_websocket_url(format, expected_encoding):
    rasa_language = "en"
    tts_engine = DeepgramTTS.from_config_dict(
        config={}, rasa_language=rasa_language, format=format
    )
    config = tts_engine.get_default_config(rasa_language)
    url = tts_engine.get_websocket_url(config)

    assert f"encoding={expected_encoding}" in url
    assert f"sample_rate={format.sample_rate}" in url


async def test_signal_interrupt_sends_clear(mulaw_format):
    """Test that signal_interrupt sends a Clear payload to Deepgram."""
    tts_engine = DeepgramTTS.from_config_dict(
        config={}, rasa_language="en", format=mulaw_format
    )

    mock_ws = mock.AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    await tts_engine.signal_interrupt()

    clear_calls = [
        call
        for call in mock_ws.send_json.call_args_list
        if call[0][0].get("type") == "Clear"
    ]
    assert len(clear_calls) == 1


async def test_stream_audio_stops_on_cleared(mulaw_format):
    """Test that stream_audio stops yielding when it receives a Cleared message."""
    tts_engine = DeepgramTTS.from_config_dict(
        config={}, rasa_language="en", format=mulaw_format
    )

    # ws should receive 2 chunks and stop after cleared message is received
    ws_messages = [
        (WSMsgType.BINARY, b"\x00\x01\x02\x03"),
        (WSMsgType.BINARY, b"\x00\x01\x02\x03"),
        (WSMsgType.TEXT, b'{"type": "Cleared"}'),
        (WSMsgType.BINARY, b"\x00\x01\x02\x03"),
    ]

    class MockWebSocket:
        def __init__(self) -> None:
            self.closed = False
            self.send_json = mock.AsyncMock()

        def __aiter__(self):
            return self._generate()

        async def _generate(self):
            for msg_type, msg_data in ws_messages:
                msg = mock.MagicMock()
                msg.type = msg_type
                msg.data = msg_data
                yield msg

    tts_engine.ws = MockWebSocket()

    received_chunks = [chunk async for chunk in tts_engine.stream_audio()]

    assert len(received_chunks) == 2
