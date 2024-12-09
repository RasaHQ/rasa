import asyncio
import difflib

import pytest

from rasa.core.channels.voice_stream.asr.asr_event import (
    NewTranscript,
    UserIsSpeaking,
)
from rasa.core.channels.voice_stream.asr.azure import AzureASR
from rasa.core.channels.voice_stream.audio_bytes import HERTZ
from rasa.core.channels.voice_stream.util import (
    generate_silence,
    read_wav_to_rasa_audio_bytes,
)


async def test_transcription(audio_data_path: str):
    rasa_audio_bytes = read_wav_to_rasa_audio_bytes(audio_data_path + "/01.wav")
    rasa_audio_bytes += generate_silence(2)
    step_size = 1024
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = AzureASR()

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
    custom_region = "local_endpoint.myurl.com"
    default_config = AzureASR.get_default_config()
    config = {"speech_region": custom_region}
    asr_engine = AzureASR.from_config_dict(config)
    assert asr_engine.config.speech_region == custom_region
    assert asr_engine.config.language == default_config.language


def test_configurating_language():
    custom_language = "es"
    config = {"language": custom_language}
    asr_engine = AzureASR.from_config_dict(config)
    assert asr_engine.config.language == custom_language
    assert (
        asr_engine.config.speech_region == AzureASR.get_default_config().speech_region
    )


def test_configuration_addioinal_attributes():
    config = {"testingXYZ@@": "@@"}
    with pytest.raises(TypeError):
        AzureASR.from_config_dict(config)
