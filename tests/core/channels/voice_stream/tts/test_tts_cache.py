from typing import Tuple

from rasa.core.channels.voice_stream.audio_bytes import (
    MULAW_8KHZ,
    AudioFormat,
    RasaAudioBytes,
)
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache


def create_fake_audio_byte_transcript_pair(
    s: str, format: AudioFormat = MULAW_8KHZ
) -> Tuple[RasaAudioBytes, str]:
    return RasaAudioBytes(s.encode("utf-8"), format=format), f"test_{s}"


def test_create_fake_audio_byte_transcript_pair():
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair("1")
    rasa_audio_bytes_2, transcript_2 = create_fake_audio_byte_transcript_pair("2")
    rasa_audio_bytes_2_again, transcript_2_again = (
        create_fake_audio_byte_transcript_pair("2")
    )

    assert rasa_audio_bytes != rasa_audio_bytes_2
    assert transcript != transcript_2
    assert rasa_audio_bytes_2 == rasa_audio_bytes_2_again
    assert transcript_2 == transcript_2_again


def test_cache_storage_and_retrieval(mulaw_format: AudioFormat):
    fmt = mulaw_format
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair(
        "1", format=fmt
    )

    cache = TTSCache(50)

    assert cache.get(transcript, fmt) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript, fmt) == rasa_audio_bytes


def test_cache_can_be_deactivated(mulaw_format: AudioFormat):
    fmt = mulaw_format
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair(
        "1", format=fmt
    )

    cache = TTSCache(0)
    assert cache.get(transcript, fmt) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript, fmt) is None


def test_cache_honors_removes_at_maxsize(mulaw_format: AudioFormat):
    fmt = mulaw_format
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair(
        "1", format=fmt
    )
    rasa_audio_bytes_2, transcript_2 = create_fake_audio_byte_transcript_pair(
        "2", format=fmt
    )

    cache = TTSCache(1)
    assert cache.get(transcript, fmt) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript, fmt) == rasa_audio_bytes

    assert cache.get(transcript_2, fmt) is None
    cache.put(transcript_2, rasa_audio_bytes_2)

    assert cache.get(transcript, fmt) is None
    assert cache.get(transcript_2, fmt) == rasa_audio_bytes_2


def test_cache_honors_usage_order(mulaw_format: AudioFormat):
    fmt = mulaw_format
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair(
        "1", format=fmt
    )
    rasa_audio_bytes_2, transcript_2 = create_fake_audio_byte_transcript_pair(
        "2", format=fmt
    )
    rasa_audio_bytes_3, transcript_3 = create_fake_audio_byte_transcript_pair(
        "3", format=fmt
    )

    cache = TTSCache(2)
    assert cache.get(transcript, fmt) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript, fmt) == rasa_audio_bytes

    assert cache.get(transcript_2, fmt) is None
    cache.put(transcript_2, rasa_audio_bytes_2)
    assert cache.get(transcript_2, fmt) == rasa_audio_bytes_2

    cache.get(transcript, fmt)

    cache.put(transcript_3, rasa_audio_bytes_3)

    assert cache.get(transcript, fmt) == rasa_audio_bytes
    assert cache.get(transcript_2, fmt) is None
    assert cache.get(transcript_3, fmt) == rasa_audio_bytes_3
