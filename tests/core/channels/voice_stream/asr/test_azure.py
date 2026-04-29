from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.asr_engine import ASRLanguageMapEntry
from rasa.core.channels.voice_stream.asr.asr_event import NewTranscript, UserIsSpeaking
from rasa.core.channels.voice_stream.asr.azure import AzureASR, AzureASRConfig
from rasa.core.channels.voice_stream.audio_bytes import (
    L16_24KHZ,
    L16_48KHZ,
    MULAW_8KHZ,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.tts_engine import TTSConfigError
from rasa.exceptions import MissingDependencyException
from rasa.shared.constants import AZURE_SPEECH_API_KEY_ENV_VAR
from rasa.shared.exceptions import ConnectionException, ProviderClientValidationError

# Keys of AzureASR.connect _FORMAT_MAP (all formats supported by Azure ASR).
AZURE_ASR_SUPPORTED_FORMATS = (MULAW_8KHZ, L16_24KHZ, L16_48KHZ)


async def test_environment_validation_raises_when_api_key_missing(
    mulaw_format, monkeypatch
) -> None:
    """AzureASR raises when the Azure Speech API key env var is unset."""
    monkeypatch.delenv(AZURE_SPEECH_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(ProviderClientValidationError) as e:
        AzureASR(
            rasa_language="en",
            format=mulaw_format,
            config=AzureASRConfig(speech_region="eastus"),
        )
    assert e.match(AZURE_SPEECH_API_KEY_ENV_VAR)
    assert e.match("ASR Engine AzureASR")


async def test_environment_validation_raises_when_azure_speech_package_missing(
    mulaw_format, monkeypatch
) -> None:
    """AzureASR raises MissingDependencyException when the SDK cannot be imported."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    monkeypatch.setattr(
        "importlib.import_module",
        mock.MagicMock(side_effect=ImportError),
    )
    with pytest.raises(MissingDependencyException) as e:
        AzureASR(
            rasa_language="en",
            format=mulaw_format,
            config=AzureASRConfig(speech_region="eastus"),
        )
    assert e.match("ASR Engine AzureASR")
    assert e.match(AzureASR.required_packages[0])


async def test_configurating_endpoint(mulaw_format):
    custom_region = "germanywestcentral"
    config = {"speech_region": custom_region}
    asr_engine = AzureASR.from_config_dict(
        config, rasa_language="en", format=mulaw_format
    )
    assert asr_engine.config.speech_region == custom_region


async def test_configurating_language(mulaw_format):
    config = {
        "speech_host": "custom.host.url",
        "language_map": {
            "en": {"language": "en-US"},
            "es": {"language": "es-ES"},
        },
    }
    asr_engine = AzureASR.from_config_dict(
        config=config,
        rasa_language="es",
        additional_languages=["en"],
        format=mulaw_format,
    )
    assert asr_engine.current_language_config.engine_language_key == "es-ES"


async def test_configuration_additional_attributes(mulaw_format):
    config = {"testingXYZ@@": "@@"}
    with pytest.warns(UserWarning, match="testingXYZ@@"):
        engine = AzureASR.from_config_dict(
            config, rasa_language="en", format=mulaw_format
        )
    assert engine is not None


@pytest.mark.parametrize("format", AZURE_ASR_SUPPORTED_FORMATS)
async def test_configuration_format(format):
    config = {"speech_region": "eastus"}
    asr_engine = AzureASR.from_config_dict(
        config=config, rasa_language="en", format=format
    )
    assert asr_engine.audio_format == format


def test_get_default_config():
    """get_default_config returns config with language_map for given rasa_language."""
    config = AzureASR.get_default_config("en")
    assert config.language_map is not None
    assert "en" in config.language_map
    assert config.language_map["en"] == ASRLanguageMapEntry(language="en-US")


@pytest.mark.parametrize("audio_format", AZURE_ASR_SUPPORTED_FORMATS)
async def test_from_config_dict_returns_azure_asr(audio_format):
    """from_config_dict instantiates AzureASR with config for each supported format."""
    config = {"speech_region": "westus2"}
    asr = AzureASR.from_config_dict(
        config=config,
        format=audio_format,
        rasa_language="en",
        additional_languages=["es"],
    )
    assert isinstance(asr, AzureASR)
    assert asr.name() == "azure"
    assert asr.audio_format == audio_format
    assert asr.config.speech_region == "westus2"
    assert asr.config.speech_host is None
    assert asr.config.speech_endpoint is None
    assert asr.config.keep_alive_interval == 5
    assert asr.config.language_map is not None
    assert asr.config.language_map["en"] == ASRLanguageMapEntry(language="en-US")
    assert asr.current_language_config.rasa_language_key == "en"
    assert asr.current_language_config.model is None


async def test_connect_raises_when_no_region_host_or_endpoint(
    mulaw_format, monkeypatch
):
    """connect raises TTSConfigError when none of region/host/endpoint set."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(),
    )
    with pytest.raises(TTSConfigError) as exc_info:
        await asr.connect()
    assert "speech_region" in str(exc_info.value)
    assert "speech_host" in str(exc_info.value)
    assert "speech_endpoint" in str(exc_info.value)


async def test_connect_raises_for_unsupported_audio_format(monkeypatch):
    """connect raises TTSConfigError for audio format not in _FORMAT_MAP."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    # Use a format that is not MULAW_8KHZ, L16_24KHZ, L16_48KHZ - we need to pass
    # a format that has no mapping. The module only maps those three; any other
    # AudioFormat would yield None from _FORMAT_MAP.get().
    from rasa.core.channels.voice_stream.audio_bytes import (
        AudioEncoding,
        AudioFormat,
    )

    unsupported = AudioFormat(
        encoding=AudioEncoding.LINEAR, sample_rate=16000, bit_depth=16
    )
    asr = AzureASR(
        rasa_language="en",
        format=unsupported,
        config=AzureASRConfig(speech_region="eastus"),
    )
    with pytest.raises(TTSConfigError) as exc_info:
        await asr.connect()
    assert "not supported by Azure ASR" in str(exc_info.value)


async def test_connect_success_with_mocked_sdk(
    mulaw_format, monkeypatch, azure_sdk_mocks
):
    """connect sets up recognizer and stream when config and format are valid."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    await asr.connect()

    assert asr.stream is azure_sdk_mocks.stream
    assert asr.speech_recognizer is azure_sdk_mocks.recognizer
    assert asr.is_recognizing is True
    azure_sdk_mocks.recognizer.recognized.connect.assert_called_once_with(
        asr.fill_queue
    )
    azure_sdk_mocks.recognizer.recognizing.connect.assert_called_once_with(
        asr.signal_user_is_speaking
    )
    azure_sdk_mocks.recognizer.start_continuous_recognition_async.assert_called_once()


async def test_close_connection_raises_when_not_connected(mulaw_format, monkeypatch):
    """close_connection raises ConnectionException when speech_recognizer is None."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    assert asr.speech_recognizer is None
    with pytest.raises(ConnectionException, match="Websocket not connected"):
        await asr.close_connection()


async def test_close_connection_calls_stop(mulaw_format, monkeypatch):
    """close_connection calls stop_continuous_recognition_async when connected."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    asr.speech_recognizer = mock.MagicMock()
    # SDK calls stop_continuous_recognition_async() without await; use MagicMock
    # to avoid "coroutine never awaited" warning
    asr.speech_recognizer.stop_continuous_recognition_async = mock.MagicMock(
        return_value=None
    )
    await asr.close_connection()
    asr.speech_recognizer.stop_continuous_recognition_async.assert_called_once()


async def test_signal_audio_done_sets_flag(mulaw_format, monkeypatch):
    """signal_audio_done sets is_recognizing to False."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    asr.is_recognizing = True
    await asr.signal_audio_done()
    assert asr.is_recognizing is False


async def test_rasa_audio_bytes_to_engine_bytes(mulaw_format, monkeypatch):
    """rasa_audio_bytes_to_engine_bytes returns chunk.data."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    chunk = RasaAudioBytes(b"abc\x00\x01", mulaw_format)
    assert asr.rasa_audio_bytes_to_engine_bytes(chunk) == b"abc\x00\x01"


async def test_send_audio_chunks_raises_when_not_connected(mulaw_format, monkeypatch):
    """send_audio_chunks raises ConnectionException when speech_recognizer is None."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    chunk = RasaAudioBytes(b"data", mulaw_format)
    with pytest.raises(ConnectionException, match="ASR not connected"):
        await asr.send_audio_chunks(chunk)


async def test_send_audio_chunks_raises_when_stream_none(mulaw_format, monkeypatch):
    """send_audio_chunks raises when stream is None (recognizer set but stream not)."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    asr.speech_recognizer = mock.MagicMock()
    asr.stream = None
    chunk = RasaAudioBytes(b"data", mulaw_format)
    with pytest.raises(ConnectionException, match="ASR not connected"):
        await asr.send_audio_chunks(chunk)


async def test_send_audio_chunks_writes_to_stream(mulaw_format, monkeypatch):
    """send_audio_chunks converts chunk and writes to stream."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    asr.speech_recognizer = mock.MagicMock()
    asr.stream = mock.MagicMock()
    chunk = RasaAudioBytes(b"audio_data", mulaw_format)
    await asr.send_audio_chunks(chunk)
    asr.stream.write.assert_called_once_with(b"audio_data")


async def test_stream_asr_events_raises_when_not_connected(mulaw_format, monkeypatch):
    """stream_asr_events raises ConnectionException when speech_recognizer is None."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    with pytest.raises(ConnectionException, match="Websocket not connected"):
        async for _ in asr.stream_asr_events():
            pass


async def test_stream_asr_events_yields_events_and_handles_timeout(
    mulaw_format, monkeypatch
):
    """stream_asr_events yields ASREvents from queue and exits when done."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    asr.speech_recognizer = mock.MagicMock()
    asr.is_recognizing = False  # so loop exits after queue is drained
    asr.queue.put_nowait(NewTranscript("hello"))
    asr.queue.put_nowait(NewTranscript("world"))
    collected = []
    async for e in asr.stream_asr_events():
        collected.append(e)
    assert len(collected) == 2
    assert collected[0].text == "hello"
    assert collected[1].text == "world"


async def test_engine_event_to_asr_event_speech_result_returns_new_transcript(
    mulaw_format, monkeypatch
):
    """engine_event_to_asr_event returns NewTranscript for SDK recognition event."""

    # Create mock classes so isinstance(e, SpeechRecognitionEventArgs) and
    # isinstance(e.result, SpeechRecognitionResult) pass inside the method
    class MockSpeechRecognitionResult:
        def __init__(self, text: str = ""):
            self.text = text

    class MockSpeechRecognitionEventArgs:
        def __init__(self, text: str = ""):
            self.result = MockSpeechRecognitionResult(text=text)

    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.SpeechRecognitionEventArgs",
        MockSpeechRecognitionEventArgs,
    )
    monkeypatch.setattr(
        "azure.cognitiveservices.speech.SpeechRecognitionResult",
        MockSpeechRecognitionResult,
    )
    mock_event = MockSpeechRecognitionEventArgs("recognized text")
    out = asr.engine_event_to_asr_event(mock_event)

    assert isinstance(out, NewTranscript)
    assert out.text == "recognized text"


async def test_engine_event_to_asr_event_passthrough_for_asr_event(
    mulaw_format, monkeypatch
):
    """engine_event_to_asr_event returns ASREvent unchanged."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    evt = UserIsSpeaking("speaking")
    assert asr.engine_event_to_asr_event(evt) is evt


async def test_engine_event_to_asr_event_returns_none_for_other(
    mulaw_format, monkeypatch
):
    """engine_event_to_asr_event returns None for unknown event type."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    assert asr.engine_event_to_asr_event(mock.MagicMock()) is None


async def test_signal_user_is_speaking_fills_queue_with_user_is_speaking(
    mulaw_format, monkeypatch
):
    """signal_user_is_speaking puts UserIsSpeaking(event.result.text) via fill_queue."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    mock_fill = mock.MagicMock()
    monkeypatch.setattr(asr, "fill_queue", mock_fill)
    mock_event = mock.MagicMock()
    mock_event.result.text = "hi"
    asr.signal_user_is_speaking(mock_event)
    mock_fill.assert_called_once()
    (arg,) = mock_fill.call_args[0]
    assert isinstance(arg, UserIsSpeaking)
    assert arg.text == "hi"


async def test_fill_queue_schedules_put_on_main_loop(mulaw_format, monkeypatch):
    """fill_queue uses main_loop.call_soon_threadsafe to put event in queue."""
    monkeypatch.setenv(AZURE_SPEECH_API_KEY_ENV_VAR, "key")
    asr = AzureASR(
        rasa_language="en",
        format=mulaw_format,
        config=AzureASRConfig(speech_region="eastus"),
    )
    mock_soon = mock.MagicMock()
    monkeypatch.setattr(asr.main_loop, "call_soon_threadsafe", mock_soon)
    evt = NewTranscript("test")
    asr.fill_queue(evt)
    mock_soon.assert_called_once()
    # First arg is put_nowait, second is the event
    assert mock_soon.call_args[0][0] == asr.queue.put_nowait
    assert mock_soon.call_args[0][1] == evt
