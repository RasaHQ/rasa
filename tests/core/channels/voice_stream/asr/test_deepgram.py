import difflib

import pytest

from rasa.core.channels.voice_stream.asr.asr_event import NewTranscript
from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR
from rasa.core.channels.voice_stream.util import read_wav_to_rasa_audio_bytes


async def test_transcription(audio_data_path: str):
    rasa_audio_bytes = read_wav_to_rasa_audio_bytes(audio_data_path + "/01.wav")
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = DeepgramASR()

    await asr_engine.connect()
    await asr_engine.send_audio_chunks(rasa_audio_bytes)
    await asr_engine.signal_audio_done()
    events = []
    async for event in asr_engine.stream_asr_events():
        events.append(event)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, NewTranscript)
    match = difflib.SequenceMatcher(None, event.text, transcript)
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
