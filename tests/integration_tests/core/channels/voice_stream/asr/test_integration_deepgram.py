from rasa.core.channels.voice_stream.asr.asr_event import UserIsSpeaking
from rasa.core.channels.voice_stream.asr.deepgram import DeepgramASR, DeepgramASRConfig
from rasa.core.channels.voice_stream.audio_bytes import MULAW_8KHZ
from tests.core.channels.voice_stream.asr import (
    run_single_utterance_transcription,
    run_transcription,
)


def get_deepgram_asr():
    return DeepgramASR(
        rasa_language="en",
        format=MULAW_8KHZ,
    )


async def test_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = get_deepgram_asr()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01_noisy.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = get_deepgram_asr()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription_without_utterance_end(audio_data_path: str):
    """Test that the utterance end feature makes a difference."""
    audio_path = audio_data_path + "/02_noisy2.wav"
    transcript = open(audio_data_path + "/02.txt").read()

    # this works fine
    asr_engine = get_deepgram_asr()
    await run_single_utterance_transcription(audio_path, transcript, asr_engine)

    # now we deactivate utterance_end detection and will not get a finalized transcript
    asr_engine = DeepgramASR(
        rasa_language="en",
        format=MULAW_8KHZ,
        config=DeepgramASRConfig(utterance_end_ms=0, endpointing=600),
    )

    events = await run_transcription(audio_path, asr_engine)

    assert len(events) > 2
    assert all([isinstance(event, UserIsSpeaking) for event in events])
