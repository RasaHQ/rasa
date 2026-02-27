from rasa.core.channels.voice_stream.asr.azure import AzureASR, AzureASRConfig
from rasa.core.channels.voice_stream.audio_bytes import MULAW_8KHZ
from tests.core.channels.voice_stream.asr import (
    run_single_utterance_transcription,
)


def get_azure_asr():
    return AzureASR(
        rasa_language="en",
        format=MULAW_8KHZ,
        config=AzureASRConfig(
            speech_region="germanywestcentral",
        ),
    )


async def test_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = get_azure_asr()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)


async def test_noisy_transcription(audio_data_path: str):
    audio_path = audio_data_path + "/01_noisy.wav"
    transcript = open(audio_data_path + "/01.txt").read()
    asr_engine = get_azure_asr()

    await run_single_utterance_transcription(audio_path, transcript, asr_engine)
