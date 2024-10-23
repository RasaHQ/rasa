from typing import Tuple

from rasa.core.channels.voice_stream.audio_bytes import RasaAudioBytes
from rasa.core.channels.voice_stream.tts.tts_cache import TTSCache


def create_fake_audio_byte_transcript_pair(s: str) -> Tuple[RasaAudioBytes, str]:
    return RasaAudioBytes(s.encode("utf-8")), f"test_{s}"


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


def test_cache_storage_and_retrieval():
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair("1")

    cache = TTSCache(50)

    assert cache.get(transcript) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript) == rasa_audio_bytes


def test_cache_can_be_deactivated():
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair("1")

    cache = TTSCache(0)
    assert cache.get(transcript) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript) is None


def test_cache_honors_removes_at_maxsize():
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair("1")
    rasa_audio_bytes_2, transcript_2 = create_fake_audio_byte_transcript_pair("2")

    cache = TTSCache(1)
    assert cache.get(transcript) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript) == rasa_audio_bytes

    assert cache.get(transcript_2) is None
    cache.put(transcript_2, rasa_audio_bytes_2)

    assert cache.get(transcript) is None
    assert cache.get(transcript_2) == rasa_audio_bytes_2


def test_cache_honors_usage_order():
    rasa_audio_bytes, transcript = create_fake_audio_byte_transcript_pair("1")
    rasa_audio_bytes_2, transcript_2 = create_fake_audio_byte_transcript_pair("2")
    rasa_audio_bytes_3, transcript_3 = create_fake_audio_byte_transcript_pair("3")

    cache = TTSCache(2)
    assert cache.get(transcript) is None
    cache.put(transcript, rasa_audio_bytes)
    assert cache.get(transcript) == rasa_audio_bytes

    assert cache.get(transcript_2) is None
    cache.put(transcript_2, rasa_audio_bytes_2)
    assert cache.get(transcript_2) == rasa_audio_bytes_2

    cache.get(transcript)

    cache.put(transcript_3, rasa_audio_bytes_3)

    assert cache.get(transcript) == rasa_audio_bytes
    assert cache.get(transcript_2) is None
    assert cache.get(transcript_3) == rasa_audio_bytes_3
