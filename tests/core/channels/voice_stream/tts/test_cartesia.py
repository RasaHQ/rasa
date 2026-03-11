import base64
from unittest import mock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.audio_bytes import L16_24KHZ, L16_48KHZ, MULAW_8KHZ
from rasa.core.channels.voice_stream.tts.cartesia import CartesiaTTS
from rasa.core.channels.voice_stream.tts.tts_engine import TTSError
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation(mulaw_format):
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            CartesiaTTS(rasa_language="en", format=mulaw_format)
        assert e.match(CartesiaTTS.required_env_vars[0])
        assert e.match("TTS Engine CartesiaTTS")


# TODO: Cartesia has stopped sending Status 400 for invalid requests
# @pytest.mark.parametrize(
#     "bad_config",
#     [
#         CartesiaTTSConfig.from_dict({"model_id": "nonexistent"}),
#         CartesiaTTSConfig.from_dict({"voice": "non_existent_voice"}),
#     ],
# )
# async def test_synthesis_error(bad_config):
#     tts_engine = CartesiaTTS()
#     text = "Hello there!"
#     with pytest.raises(TTSError):
#         async for chunk in tts_engine.synthesize(text, bad_config):
#             pass


async def test_synthesis_bad_api_key(monkeypatch: MonkeyPatch, mulaw_format):
    monkeypatch.setenv("CARTESIA_API_KEY", "bad_key")
    tts_engine = CartesiaTTS(rasa_language="en", format=mulaw_format)
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_default_config():
    config = CartesiaTTS.get_default_config()
    assert "en" in config.language_map
    assert config.language_map["en"].language == "en"
    assert config.language_map["en"].voice == "f786b574-daa5-4673-aa0c-cbe3e8534c02"
    assert config.model_id == "sonic-3"
    assert config.version == "2025-04-16"


async def test_tts_session_sharing(mulaw_format):
    tts_engine = CartesiaTTS(rasa_language="en", format=mulaw_format)
    tts_engine_2 = CartesiaTTS(rasa_language="en", format=mulaw_format)
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
    tts_engine = CartesiaTTS.from_config_dict(
        config={}, rasa_language="en", format=format
    )
    assert tts_engine.audio_format == format


@pytest.mark.parametrize(
    "format, expected_encoding",
    [
        (MULAW_8KHZ, "pcm_mulaw"),
        (L16_24KHZ, "pcm_s16le"),
        (L16_48KHZ, "pcm_s16le"),
    ],
)
async def test_send_text_chunk_payload(format, expected_encoding):
    tts_engine = CartesiaTTS.from_config_dict(
        config={}, rasa_language="en", format=format
    )

    # Mock the WebSocket and ensure it's not closed
    mock_ws = mock.AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    text = "Hello, Cartesia!"
    await tts_engine.send_text_chunk(text)

    tts_engine.ws.send_json.assert_called_once()
    payload = tts_engine.ws.send_json.call_args[0][0]

    assert payload["output_format"]["encoding"] == expected_encoding
    assert payload["output_format"]["sample_rate"] == format.sample_rate
    assert payload["transcript"] == text


async def test_signal_interrupt_sends_cancel(mulaw_format):
    """Test that signal_interrupt sends a cancel payload to Cartesia."""
    tts_engine = CartesiaTTS.from_config_dict(
        config={}, rasa_language="en", format=mulaw_format
    )

    mock_ws = mock.AsyncMock()
    mock_ws.closed = False
    tts_engine.ws = mock_ws

    await tts_engine.signal_interrupt()

    cancel_calls = [
        call
        for call in mock_ws.send_json.call_args_list
        if call[0][0].get("cancel") is True
    ]
    assert len(cancel_calls) == 1


async def test_stream_audio_stops_on_done(mulaw_format):
    """Test that stream_audio stops yielding when it receives a done message."""
    tts_engine = CartesiaTTS.from_config_dict(
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
