import asyncio
import difflib
from unittest import mock

import pytest

from rasa.core.channels.voice_stream.asr.asr_event import (
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.audio_bytes import HERTZ
from rasa.core.channels.voice_stream.util import (
    generate_silence,
    read_wav_to_rasa_audio_bytes,
)
from rasa.shared.exceptions import ProviderClientValidationError


async def test_environment_validation():
    # no api key set
    with mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ProviderClientValidationError) as e:
            DeepgramASR()
        assert e.match(DeepgramASR.required_env_vars[0])
        assert e.match("ASR Engine DeepgramASR")


async def test_transcription(audio_data_path: str):
    rasa_audio_bytes = read_wav_to_rasa_audio_bytes(audio_data_path + "/01.wav")
    rasa_audio_bytes += generate_silence(2.0)
    transcript = open(audio_data_path + "/01.txt").read()
    step_size = 1024
    asr_engine = DeepgramASR()

    await asr_engine.connect()
    offset = 0
    while offset < len(rasa_audio_bytes):
        await asr_engine.send_audio_chunks(
            rasa_audio_bytes[offset : offset + step_size]
        )
        offset += step_size
        await asyncio.sleep(step_size / HERTZ)
    await asr_engine.signal_audio_done()

    events = []
    async for event in asr_engine.stream_asr_events():
        events.append(event)

    assert len(events) > 2
    assert all([isinstance(event, UserIsSpeaking) for event in events[:-1]])
    assert isinstance(events[-1], NewTranscript)
    match = difflib.SequenceMatcher(None, events[-1].text, transcript)
    assert match.ratio() > 0.75


def test_configurating_endpoint():
    custom_endpoint = "local_endpoint.myurl.com"
    default_config = DeepgramASR.get_default_config()
    config = {"endpoint": custom_endpoint}
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.endpoint == custom_endpoint
    assert asr_engine.config.endpointing == default_config.endpointing
    assert custom_endpoint in asr_engine._get_api_url()


def test_configurating_endpointing():
    custom_endpointing = 1000
    config = {"endpointing": custom_endpointing}
    default_config = DeepgramASR.get_default_config()
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.endpointing == custom_endpointing
    assert asr_engine.config.endpoint == default_config.endpoint
    assert f"endpointing={custom_endpointing}" in asr_engine._get_query_params()


def test_configurating_language():
    custom_language = "es"
    config = {"language": custom_language}
    asr_engine = DeepgramASR.from_config_dict(config)
    assert asr_engine.config.language == custom_language
    assert f"language={custom_language}" in asr_engine._get_query_params()


def test_configuration_addioinal_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(TypeError):
        DeepgramASR.from_config_dict(config)
