from rasa.core.channels.voice_stream.asr.azure import AzureASR
from tests.core.channels.voice_stream.asr import (
    run_single_utterance_transcription,
)


async def test_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = AzureASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01_noisy.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = AzureASR()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)
