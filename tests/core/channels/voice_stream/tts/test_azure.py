import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import azure.cognitiveservices.speech as speechsdk
import pytest
from pytest import MonkeyPatch

from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.azure import (
    AzureTTS,
    AzureTTSConfig,
    _AudioOutputCallback,
)
from rasa.core.channels.voice_stream.tts.config import StreamingConfig
from rasa.core.channels.voice_stream.tts.tts_engine import (
    TTSError,
)
from rasa.shared.constants import (
    AZURE_AD_SCOPES_ENV_VAR,
    AZURE_AD_TOKEN_ENV_VAR,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_KEY_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    AZURE_SPEECH_API_KEY_ENV_VAR,
)
from rasa.shared.exceptions import ProviderClientValidationError
from tests.core.channels.voice_stream.tts.test_tts import (
    run_single_utterance_through_tts_and_asr,
)


@pytest.fixture
async def azure_tts(monkeypatch: MonkeyPatch) -> AzureTTS:
    """Create an AzureTTS instance with mocked env var."""
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "test-key")
    return AzureTTS(rasa_language="en", format=L16_24KHZ)


async def test_prepare_response_ssml_disables_streaming(azure_tts: AzureTTS):
    """SSML content should disable streaming even when SDK is available."""
    azure_tts._synthesizer = MagicMock()
    await azure_tts.prepare_response(StreamingConfig(response_text_contains_ssml=True))
    assert azure_tts._use_streaming is False


async def test_prepare_response_no_ssml_enables_streaming(azure_tts: AzureTTS):
    """No SSML + SDK available should enable streaming."""
    azure_tts._synthesizer = MagicMock()
    azure_tts._speech_config = MagicMock()
    azure_tts._callback = MagicMock()
    azure_tts._loop = asyncio.get_running_loop()
    mock_request = MagicMock()
    with patch("rasa.core.channels.voice_stream.tts.azure.speechsdk") as mock_sdk:
        mock_sdk.SpeechSynthesisRequest.return_value = mock_request
        await azure_tts.prepare_response(
            StreamingConfig(response_text_contains_ssml=False)
        )
    assert azure_tts._use_streaming is True


async def test_prepare_response_no_sdk_disables_streaming(azure_tts: AzureTTS):
    """Without SDK synthesizer, streaming should be disabled."""
    azure_tts._synthesizer = None
    await azure_tts.prepare_response(StreamingConfig(response_text_contains_ssml=False))
    assert azure_tts._use_streaming is False


async def test_prepare_response_default_no_ssml(azure_tts: AzureTTS):
    """No config passed should default to streaming."""
    azure_tts._synthesizer = MagicMock()
    azure_tts._speech_config = MagicMock()
    azure_tts._callback = MagicMock()
    azure_tts._loop = asyncio.get_running_loop()
    mock_request = MagicMock()
    with patch("rasa.core.channels.voice_stream.tts.azure.speechsdk") as mock_sdk:
        mock_sdk.SpeechSynthesisRequest.return_value = mock_request
        await azure_tts.prepare_response()
    assert azure_tts._use_streaming is True


async def test_send_text_chunk_buffers_in_rest_mode(azure_tts: AzureTTS):
    """send_text_chunk buffers text when not streaming."""
    azure_tts._use_streaming = False
    azure_tts._text_buffer = []

    await azure_tts.send_text_chunk("Hello")
    await azure_tts.send_text_chunk(" world")
    assert azure_tts._text_buffer == ["Hello", " world"]


async def test_send_text_chunk_writes_to_sdk_when_streaming(azure_tts: AzureTTS):
    """send_text_chunk writes directly to SDK input stream when streaming."""
    azure_tts._use_streaming = True
    mock_request = MagicMock()
    mock_stream = MagicMock()
    mock_request.input_stream = mock_stream
    azure_tts._tts_request = mock_request

    await azure_tts.send_text_chunk("Hello")
    await azure_tts.send_text_chunk(" world")

    assert mock_stream.write.call_count == 2
    mock_stream.write.assert_any_call("Hello")
    mock_stream.write.assert_any_call(" world")
    # Should NOT buffer
    assert azure_tts._text_buffer == []


async def test_signal_text_done_when_not_streaming(
    azure_tts: AzureTTS, mulaw_format: AudioFormat
):
    """Test signal_text_done when not streaming.

    It should clear the text buffer and put audio chunks to the audio queue.
    """
    azure_tts._use_streaming = False
    azure_tts._text_buffer = ["Hello", " world"]
    azure_tts._audio_queue = asyncio.Queue()

    audio_chunks = [
        RasaAudioBytes(b"audio1", mulaw_format),
        RasaAudioBytes(b"audio2", mulaw_format),
    ]

    async def mock_rest(text, config=None):
        for chunk in audio_chunks:
            yield chunk

    with patch.object(azure_tts, "_synthesize_rest", side_effect=mock_rest):
        await azure_tts.signal_text_done()

    assert azure_tts._text_buffer == []

    chunk1 = await azure_tts._audio_queue.get()
    assert chunk1 == audio_chunks[0]
    chunk2 = await azure_tts._audio_queue.get()
    assert chunk2 == audio_chunks[1]

    # assert that None is put on the audio queue to indicate end of stream
    sentinel = await azure_tts._audio_queue.get()
    assert sentinel is None


async def test_signal_text_done_when_streaming(azure_tts: AzureTTS):
    """Tests signal_text_done when streaming.

    It should wait for the tts future and close the stream once it is done.
    """
    azure_tts._use_streaming = True
    azure_tts._audio_queue = asyncio.Queue()
    azure_tts._loop = asyncio.get_running_loop()

    mock_request = MagicMock()
    mock_stream = MagicMock()
    mock_request.input_stream = mock_stream
    azure_tts._tts_request = mock_request

    mock_future = MagicMock()
    mock_future.get = MagicMock(return_value=MagicMock())
    azure_tts._tts_future = mock_future

    await azure_tts.signal_text_done()

    mock_stream.close.assert_called_once()

    # assert that None is put on the audio queue to indicate end of stream
    sentinel = await azure_tts._audio_queue.get()
    assert sentinel is None


async def test_stream_audio_reads_from_queue(
    azure_tts: AzureTTS, mulaw_format: AudioFormat
):
    """stream_audio always reads from queue."""

    async def populate_queue():
        await asyncio.sleep(0.01)
        await azure_tts._audio_queue.put(RasaAudioBytes(b"chunk1", mulaw_format))
        await azure_tts._audio_queue.put(RasaAudioBytes(b"chunk2", mulaw_format))
        await azure_tts._audio_queue.put(None)

    task = asyncio.create_task(populate_queue())

    chunks = []
    async for chunk in azure_tts.stream_audio():
        chunks.append(chunk)

    await task
    assert len(chunks) == 2
    assert chunks[0] == RasaAudioBytes(b"chunk1", mulaw_format)
    assert chunks[1] == RasaAudioBytes(b"chunk2", mulaw_format)


def test_audio_callback_write_pushes_to_queue(mulaw_format: AudioFormat):
    """Callback write() should push audio to the asyncio queue."""
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    callback = _AudioOutputCallback(loop, queue, mulaw_format)
    assert callback.queue is queue

    audio_data = b"\xff\x00\x01\x02"
    # Simulate SDK calling write from its thread
    # We run it directly here for simplicity
    result = callback.write(memoryview(audio_data))

    assert result == len(audio_data)
    # Since we called from same thread as loop owner, use loop to drain
    loop.run_until_complete(asyncio.sleep(0))
    assert not queue.empty()
    chunk = loop.run_until_complete(queue.get())
    assert chunk == RasaAudioBytes(audio_data, mulaw_format)
    loop.close()


def test_audio_callback_close_is_noop(mulaw_format: AudioFormat):
    """Assert that closing the audio callback does not put anything on audio queue."""
    loop = asyncio.new_event_loop()
    queue = asyncio.Queue()
    callback = _AudioOutputCallback(loop, queue, mulaw_format)

    callback.close()

    loop.run_until_complete(asyncio.sleep(0))
    assert queue.empty()
    loop.close()


async def test_synthesize_always_uses_rest(
    azure_tts: AzureTTS, mulaw_format: AudioFormat
):
    """synthesize() should always use REST (for template responses)."""
    audio_chunks = [RasaAudioBytes(b"audio", mulaw_format)]

    async def mock_rest(text: str, config=None):
        for chunk in audio_chunks:
            yield chunk

    with patch.object(azure_tts, "_synthesize_rest", side_effect=mock_rest):
        result = []
        async for chunk in azure_tts.synthesize("Hello"):
            result.append(chunk)

    assert result == audio_chunks


async def test_close_connection_clears_sdk(azure_tts: AzureTTS):
    azure_tts._synthesizer = MagicMock()
    azure_tts._speech_config = MagicMock()

    await azure_tts.close_connection()
    assert azure_tts._synthesizer is None
    assert azure_tts._speech_config is None


async def test_close_connection_noop_when_no_sdk(azure_tts: AzureTTS):
    azure_tts._synthesizer = None
    azure_tts._speech_config = None
    # Should not raise
    await azure_tts.close_connection()


ALL_AZURE_ENV_VARS = [
    AZURE_API_KEY_ENV_VAR,
    AZURE_AD_TOKEN_ENV_VAR,
    AZURE_API_BASE_ENV_VAR,
    AZURE_API_VERSION_ENV_VAR,
    AZURE_API_TYPE_ENV_VAR,
    AZURE_AD_SCOPES_ENV_VAR,
    AZURE_SPEECH_API_KEY_ENV_VAR,
]


@pytest.mark.asyncio
async def test_environment_validation(
    monkeypatch: MonkeyPatch, mulaw_format: AudioFormat
):
    # no api key set
    for env_Var in ALL_AZURE_ENV_VARS:
        monkeypatch.delenv(env_Var, raising=False)

    with pytest.raises(ProviderClientValidationError) as e:
        AzureTTS(rasa_language="en", format=mulaw_format)
    assert e.match(AzureTTS.required_env_vars[0])
    assert e.match("TTS Engine AzureTTS")


@pytest.mark.asyncio
async def test_synthesis_with_asr(mulaw_format: AudioFormat):
    tts_engine = AzureTTS(
        rasa_language="en",
        format=mulaw_format,
        config=AzureTTSConfig(
            speech_region="germanywestcentral",
        ),
    )
    text = "hello my name is Edgar"
    asr_engine = DeepgramASR(rasa_language="en", format=mulaw_format)
    await run_single_utterance_through_tts_and_asr(
        text, asr_engine, tts_engine, mulaw_format
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_config",
    [
        AzureTTSConfig(speech_region="nonexistent"),
        AzureTTSConfig(voice="non_existent_voice"),
    ],
)
async def test_synthesis_error(bad_config: AzureTTSConfig, mulaw_format: AudioFormat):
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text, bad_config):
            pass


@pytest.mark.asyncio
async def test_synthesis_bad_api_key(
    monkeypatch: MonkeyPatch, mulaw_format: AudioFormat
):
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "bad key")
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    text = "Hello there!"
    with pytest.raises(TTSError):
        async for chunk in tts_engine.synthesize(text):
            pass


def test_azure_default_config():
    config = AzureTTS.get_default_config("en")
    assert "en" in config.language_map
    assert config.language_map["en"].language == "en-US"
    assert config.language_map["en"].voice == "en-US-JennyNeural"
    assert config.speech_region == "eastus"


def test_tts_url_creation():
    config = AzureTTS.get_default_config("en")
    azure_tts_endpoint = AzureTTS.get_tts_endpoint(config)
    assert config.speech_region in azure_tts_endpoint
    assert azure_tts_endpoint.startswith("https://")


def test_tts_request_body(mulaw_format: AudioFormat):
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    text = "Hi there, how can I help you today?"
    request_body = AzureTTS.create_request_body(
        text, tts_engine.current_language_config
    )
    assert text in request_body
    assert tts_engine.current_language_config.voice in request_body
    assert tts_engine.current_language_config.engine_language_key in request_body


async def test_tts_headers(mulaw_format: AudioFormat):
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    headers = tts_engine.get_request_headers()
    assert "Ocp-Apim-Subscription-Key" in headers
    assert "Content-Type" in headers
    assert "X-Microsoft-OutputFormat" in headers
    assert headers["X-Microsoft-OutputFormat"] == "raw-8khz-8bit-mono-mulaw"


async def test_tts_headers_l16_24khz():
    tts_engine = AzureTTS(rasa_language="en", format=L16_24KHZ)
    headers = tts_engine.get_request_headers()
    assert headers["X-Microsoft-OutputFormat"] == "raw-24khz-16bit-mono-pcm"


async def test_tts_headers_l16_48khz():
    tts_engine = AzureTTS(rasa_language="en", format=L16_48KHZ)
    headers = tts_engine.get_request_headers()
    assert headers["X-Microsoft-OutputFormat"] == "raw-48khz-16bit-mono-pcm"


@pytest.mark.asyncio
async def test_tts_session_sharing(mulaw_format: AudioFormat):
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    tts_engine_2 = AzureTTS(rasa_language="en", format=mulaw_format)
    assert tts_engine_2.session is tts_engine.session


@pytest.mark.asyncio
async def test_synthesize_timeout(monkeypatch: MonkeyPatch, mulaw_format: AudioFormat):
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "my key")
    tts_engine = AzureTTS(rasa_language="en", format=mulaw_format)
    text = "Test timeout"

    # Mock the response to be an async context manager
    mock_response = AsyncMock()
    # Did this to avoid AttributeError: __aenter__ error
    mock_response.__aenter__.side_effect = TimeoutError("Request timed out")

    # Inject a mock client session and patch the `post` call.
    tts_engine.session = MagicMock()
    with patch.object(tts_engine.session, "post", return_value=mock_response):
        with pytest.raises(TTSError) as exc_info:
            async for chunk in tts_engine.synthesize(text):
                pass

        assert "Request timed out" in str(exc_info.value)


# ── streaming_input class attribute test ─────────────────────────────


def test_streaming_input_is_true():
    """AzureTTS should now have streaming_input=True."""
    assert AzureTTS.streaming_input is True


# ── connect method tests ──────────────────────────────────────────────


async def test_connect_initializes_sdk_objects(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should set up speech_config, synthesizer, and callback."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")

    mock_speech_config = MagicMock()
    mock_synthesizer = MagicMock()
    mock_push_stream = MagicMock()
    mock_audio_config = MagicMock()
    mock_callback = MagicMock()

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_sdk.SpeechConfig.return_value = mock_speech_config
    mock_sdk.SpeechSynthesizer.return_value = mock_synthesizer
    mock_sdk.audio.PushAudioOutputStream.return_value = mock_push_stream
    mock_sdk.audio.AudioOutputConfig.return_value = mock_audio_config
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    assert azure_tts._speech_config is mock_speech_config
    assert azure_tts._synthesizer is mock_synthesizer
    assert azure_tts._callback is mock_callback
    assert azure_tts._loop is not None


async def test_connect_sets_voice_name_on_speech_config(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should configure the voice name on the speech config."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")

    mock_speech_config = MagicMock()

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_sdk.SpeechConfig.return_value = mock_speech_config
    mock_sdk.SpeechSynthesizer.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStream.return_value = MagicMock()
    mock_sdk.audio.AudioOutputConfig.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    mock_callback = MagicMock()
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    assert (
        mock_speech_config.speech_synthesis_voice_name
        == azure_tts.current_language_config.voice
    )


async def test_connect_uses_default_ws_endpoint_when_not_configured(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should build the wss endpoint from speech_
    region when ws_endpoint is None."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")
    azure_tts.config.ws_endpoint = None
    azure_tts.config.speech_region = "westeurope"

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_speech_config = MagicMock()

    mock_sdk.SpeechConfig = mock_speech_config
    mock_sdk.SpeechSynthesizer.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStream.return_value = MagicMock()
    mock_sdk.audio.AudioOutputConfig.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    mock_callback = MagicMock()
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    mock_sdk.SpeechConfig.assert_called_once_with(
        subscription="test-key",
        endpoint=f"wss://{azure_tts.config.speech_region}.tts.speech.microsoft.com/"
        f"cognitiveservices/websocket/v2",
    )


async def test_connect_uses_custom_ws_endpoint_when_configured(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should use the custom ws_endpoint when provided."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")
    custom_endpoint = "wss://my.custom.endpoint/cognitiveservices/websocket/v2"
    azure_tts.config.ws_endpoint = custom_endpoint

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_speech_config = MagicMock()

    mock_sdk.SpeechConfig = mock_speech_config
    mock_sdk.SpeechSynthesizer.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStream.return_value = MagicMock()
    mock_sdk.audio.AudioOutputConfig.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    mock_callback = MagicMock()
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    mock_sdk.SpeechConfig.assert_called_once_with(
        subscription="test-key",
        endpoint=azure_tts.config.ws_endpoint,
    )


async def test_connect_skips_when_voice_is_missing(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should skip SDK init when no voice is configured."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")
    azure_tts.current_language_config.voice = None

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)
    await azure_tts.connect()
    mock_sdk.SpeechConfig.assert_not_called()

    assert azure_tts._synthesizer is None
    assert azure_tts._speech_config is None


async def test_connect_sets_synthesizer_to_none_on_exception(
    azure_tts: AzureTTS, monkeypatch: MonkeyPatch
):
    """connect() should set _synthesizer to None when SDK raises an exception."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_sdk.SpeechConfig.side_effect = RuntimeError("SDK init failed")

    await azure_tts.connect()
    assert azure_tts._synthesizer is None


async def test_connect_sets_loop(azure_tts: AzureTTS, monkeypatch: MonkeyPatch):
    """connect() should capture the running event loop."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_sdk.SpeechConfig.return_value = MagicMock()
    mock_sdk.SpeechSynthesizer.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStream.return_value = MagicMock()
    mock_sdk.audio.AudioOutputConfig.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    mock_callback = MagicMock()
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    assert azure_tts._loop is asyncio.get_event_loop()


@pytest.mark.parametrize(
    "audio_format",
    [
        {
            "rasa_format": MULAW_8KHZ,
            "expected_azure_format": speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw,  # noqa: E501
        },
        {
            "rasa_format": L16_24KHZ,
            "expected_azure_format": speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm,  # noqa: E501
        },
        {
            "rasa_format": L16_48KHZ,
            "expected_azure_format": speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm,  # noqa: E501
        },
    ],
)
async def test_connect_sets_audio_output_format(
    monkeypatch: MonkeyPatch, audio_format: Dict[str, Any]
):
    """connect() should call set_speech_synthesis_output_format on speech_config."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "test-key")
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "test-key")
    azure_tts = AzureTTS(rasa_language="en", format=audio_format.get("rasa_format"))

    mock_speech_config = MagicMock()

    mock_sdk = MagicMock()
    monkeypatch.setattr("rasa.core.channels.voice_stream.tts.azure.speechsdk", mock_sdk)

    mock_sdk.SpeechConfig.return_value = mock_speech_config
    mock_sdk.SpeechSynthesizer.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStream.return_value = MagicMock()
    mock_sdk.audio.AudioOutputConfig.return_value = MagicMock()
    mock_sdk.audio.PushAudioOutputStreamCallback = (
        speechsdk.audio.PushAudioOutputStreamCallback
    )

    mock_callback = MagicMock()
    monkeypatch.setattr(
        "rasa.core.channels.voice_stream.tts.azure._AudioOutputCallback",
        MagicMock(return_value=mock_callback),
    )

    await azure_tts.connect()

    mock_speech_config.set_speech_synthesis_output_format.assert_called_once_with(
        audio_format.get("expected_azure_format")
    )


# ── _get_azure_audio_format tests ────────────────────────────────────


@pytest.mark.parametrize(
    "rasa_format, expected_azure_format",
    [
        (MULAW_8KHZ, speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw),
        (L16_24KHZ, speechsdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm),
        (L16_48KHZ, speechsdk.SpeechSynthesisOutputFormat.Raw48Khz16BitMonoPcm),
    ],
)
async def test_get_azure_audio_format_returns_correct_format(
    rasa_format: AudioFormat,
    expected_azure_format: speechsdk.SpeechSynthesisOutputFormat,
    monkeypatch: MonkeyPatch,
):
    """_get_azure_audio_format should map each Rasa
    AudioFormat to the correct Azure SDK enum."""
    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "test-key")
    tts_engine = AzureTTS(rasa_language="en", format=rasa_format)
    assert tts_engine._get_azure_audio_format() == expected_azure_format


async def test_get_azure_audio_format_raises_for_unsupported_format(
    monkeypatch: MonkeyPatch,
):
    """_get_azure_audio_format should raise ValueError for
    an unsupported AudioFormat."""
    from rasa.core.channels.voice_stream.audio_bytes import AudioEncoding, AudioFormat

    monkeypatch.setenv("AZURE_SPEECH_API_KEY", "test-key")
    unsupported_format = AudioFormat(
        encoding=AudioEncoding.LINEAR,
        sample_rate=44100,
        bit_depth=16,
    )
    tts_engine = AzureTTS(rasa_language="en", format=unsupported_format)
    with pytest.raises(ValueError, match="Azure TTS does not support audio format"):
        tts_engine._get_azure_audio_format()


@pytest.mark.parametrize(
    "output_format",
    [
        MULAW_8KHZ,
        L16_24KHZ,
        L16_48KHZ,
    ],
)
async def test_configuration_format(output_format: AudioFormat):
    config = {"speech_region": "eastus"}
    tts_engine = AzureTTS.from_config_dict(
        config=config, rasa_language="en", format=output_format
    )
    assert tts_engine.audio_format == output_format
