import wave
from datetime import datetime
from pathlib import Path

import structlog

from rasa.core.channels.voice_stream.audio_bytes import AudioEncoding, RasaAudioBytes

logger = structlog.get_logger()


def _save_rasa_bytes_to_wav(audio: RasaAudioBytes, dir_name: str) -> None:
    """Save RasaAudioBytes audio to WAV file for debugging."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = Path(dir_name)
        debug_dir.mkdir(exist_ok=True)

        encoding_str = audio.format.encoding.value
        sample_rate = audio.format.sample_rate
        filename = debug_dir / f"{encoding_str}_{sample_rate}hz_{timestamp}.wav"

        # Determine WAV parameters based on audio format
        if audio.format.encoding == AudioEncoding.LINEAR:
            sampwidth = audio.format.bit_depth // 8
            comptype = "NONE"
            compname = "not compressed"
        elif audio.format.encoding == AudioEncoding.MULAW:
            sampwidth = 1  # 8-bit audio
            comptype = "ULAW"
            compname = "CCITT G.711 u-law"
        else:
            logger.error(
                "voice.debug.unsupported_encoding", encoding=audio.format.encoding
            )
            return

        with wave.open(str(filename), "wb") as wav_file:
            wav_file.setnchannels(audio.format.channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(audio.format.sample_rate)
            wav_file.setcomptype(comptype, compname)
            wav_file.writeframes(audio.data)

        logger.info(
            "voice.debug.wav_saved",
            filename=str(filename),
            size_bytes=len(audio.data),
            encoding=encoding_str,
            sample_rate=sample_rate,
        )
    except Exception as e:
        logger.error("voice.debug.save_error", error=str(e), exc_info=True)
