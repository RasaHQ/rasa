import base64
import math
import struct
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
)
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


@pytest.fixture
def set_rime_key(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("RIME_API_KEY", "test_key")


@pytest.mark.usefixtures("set_rime_key")
async def test_websocket_url_creation(mulaw_format: AudioFormat):
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)
    ws_url = tts_engine.get_websocket_url()
    assert "wss://users.rime.ai/ws2" in ws_url
    assert "speaker=cove" in ws_url
    assert "modelId=mistv2" in ws_url
    assert "lang=eng" in ws_url
    assert "audioFormat=mulaw" in ws_url
    assert f"samplingRate={mulaw_format.sample_rate}" in ws_url


@pytest.mark.usefixtures("set_rime_key")
async def test_websocket_url_with_optional_params(mulaw_format: AudioFormat):
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


@pytest.mark.usefixtures("set_rime_key")
async def test_websocket_url_with_no_text_normalization(mulaw_format: AudioFormat):
    config = RimeTTSConfig(
        model_id="mistv2",
        no_text_normalization=True,
        language_map={
            "en": {"language": "eng", "voice": "cove"},
        },
    )
    tts_engine = RimeTTS(rasa_language="en", config=config, format=mulaw_format)
    ws_url = tts_engine.get_websocket_url()
    assert "noTextNormalization=true" in ws_url


@pytest.mark.usefixtures("set_rime_key")
async def test_websocket_url_no_text_normalization_absent_by_default(
    mulaw_format: AudioFormat,
):
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)
    ws_url = tts_engine.get_websocket_url()
    assert "noTextNormalization" not in ws_url


@pytest.mark.usefixtures("set_rime_key")
def test_request_headers():
    headers = RimeTTS.get_request_headers()
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test_key"


@pytest.mark.usefixtures("set_rime_key")
async def test_signal_text_done_resets_context_id(mulaw_format: AudioFormat):
    """Test that context_id gets reset every time signal_text_done is called."""
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


@pytest.mark.usefixtures("set_rime_key")
async def test_signal_text_done_raises_error_when_ws_not_connected(
    mulaw_format: AudioFormat,
):
    """Test that signal_text_done raises TTSError when WebSocket is not connected."""
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # WebSocket is None (not connected)
    tts_engine.ws = None

    with pytest.raises(TTSError, match="WebSocket connection not established"):
        await tts_engine.signal_text_done()


@pytest.mark.usefixtures("set_rime_key")
async def test_signal_text_done_raises_error_when_ws_closed(mulaw_format: AudioFormat):
    """Test that signal_text_done raises TTSError when WebSocket is closed."""
    tts_engine = RimeTTS(rasa_language="en", format=mulaw_format)

    # Mock a closed websocket
    mock_ws = AsyncMock()
    mock_ws.closed = True
    tts_engine.ws = mock_ws

    with pytest.raises(TTSError, match="WebSocket connection not established"):
        await tts_engine.signal_text_done()


@pytest.mark.usefixtures("set_rime_key")
async def test_send_text_chunk_includes_context_id(mulaw_format: AudioFormat):
    """Test that send_text_chunk sends text with the current context_id."""
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
    "audio_format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        # We allow for this format to be sent as an input config to RimeTTS
        # Internally we will instruct rime to use 24KHz sample rate to produce audio
        # which we will later on upsample to 48KHz
        L16_48KHZ,
    ],
)
@pytest.mark.usefixtures("set_rime_key")
async def test_configuration_format(audio_format: AudioFormat):
    tts_engine = RimeTTS.from_config_dict(
        config={}, rasa_language="en", format=audio_format
    )
    assert tts_engine.audio_format == audio_format


@pytest.mark.parametrize(
    "audio_format, expected_encoding",
    [
        (MULAW_8KHZ, "mulaw"),
        (L16_24KHZ, "pcm"),
    ],
)
@pytest.mark.usefixtures("set_rime_key")
async def test_get_websocket_url(audio_format: AudioFormat, expected_encoding: str):
    tts_engine = RimeTTS.from_config_dict(
        config={}, rasa_language="en", format=audio_format
    )
    ws_url = tts_engine.get_websocket_url()

    assert f"audioFormat={expected_encoding}" in ws_url
    assert f"samplingRate={audio_format.sample_rate}" in ws_url


@pytest.mark.usefixtures("set_rime_key")
async def test_48KHz_format_uses_24kHz():
    """Test that using 48kHz resorts to 44_100Hz format."""
    tts_engine = RimeTTS.from_config_dict(
        config={}, rasa_language="en", format=L16_48KHZ
    )
    result = tts_engine.get_websocket_url()

    assert "audioFormat=pcm" in result
    assert "samplingRate=24000" in result


@pytest.mark.usefixtures("set_rime_key")
async def test_signal_interrupt_sends_clear(mulaw_format: AudioFormat):
    """Test that signal_interrupt sends a clear payload to Rime."""
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


@pytest.mark.usefixtures("set_rime_key")
async def test_stream_audio_stops_on_done(mulaw_format: AudioFormat):
    """Test that stream_audio stops yielding when it receives a done message."""
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


@pytest.fixture
def sine_audio_bytes() -> bytes:
    """Generate 1 second of a 440 Hz sine wave as 16-bit signed PCM mono at 44100 Hz.

    Format: little-endian int16, mono, 44100 Hz → 88200 bytes total.
    """
    sample_rate = 24000
    frequency = 440.0  # A4 note, Hz
    duration = 1.0  # seconds
    amplitude = 32767  # max amplitude for signed 16-bit PCM

    num_samples = int(sample_rate * duration)
    samples = [
        int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(num_samples)
    ]
    return struct.pack(f"<{num_samples}h", *samples)


@pytest.mark.usefixtures("set_rime_key")
async def test_transcode_audio_passthrough_mulaw():
    """transcode_audio returns bytes unchanged for MULAW_8KHZ (sample rate ≤ 44100)."""
    tts_engine = RimeTTS(rasa_language="en", format=MULAW_8KHZ)

    result = tts_engine.transcode_audio(b"aaaaa")

    assert result == b"aaaaa"


@pytest.mark.usefixtures("set_rime_key")
async def test_transcode_audio_passthrough_l16_24khz():
    """transcode_audio returns bytes unchanged for L16_24KHZ (sample rate ≤ 44100)."""
    tts_engine = RimeTTS(rasa_language="en", format=L16_24KHZ)

    result = tts_engine.transcode_audio(b"aaaaaaa")

    assert result == b"aaaaaaa"


@pytest.mark.usefixtures("set_rime_key")
async def test_transcode_audio_resamples_to_48khz(sine_audio_bytes: bytes):
    """transcode_audio resamples 44100 Hz PCM input to 48000 Hz for L16_48KHZ format."""
    tts_engine = RimeTTS(rasa_language="en", format=L16_48KHZ)

    result = tts_engine.transcode_audio(sine_audio_bytes)

    # Output must differ from the raw input
    assert result != sine_audio_bytes

    # After upsampling 44100 → 48000 the output has more bytes.
    # Expected: ~48000 samples × 2 bytes = 96000 bytes.
    # audioop.ratecv may be off by a few samples due to internal rounding.
    input_samples = len(sine_audio_bytes) // 2  # 44100 samples
    expected_output_samples = round(input_samples * 48000 / 24000)
    expected_output_bytes = expected_output_samples * 2  # 16-bit → 2 bytes/sample

    # within 4 samples, due to rounding error in ratecv
    assert abs(len(result) - expected_output_bytes) <= 8

    # Upsampling 44100 → 48000 Hz must produce more bytes than the input.
    assert len(result) > len(sine_audio_bytes)

    # After resampling, the audio duration should remain approximately 1 second.
    # This is due to rounding error in ratecv
    input_duration = len(sine_audio_bytes) / (24000 * 2)
    output_duration = len(result) / (48000 * 2)
    assert abs(input_duration - output_duration) < 0.01  # within 10 ms
