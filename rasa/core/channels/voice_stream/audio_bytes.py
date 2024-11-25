from typing import NewType

# a common intermediate audio byte format that acts as a common data format,
# to prevent quadratic complexity between formats of channels, asr engines,
# and tts engines
# currently corresponds to raw wave, 8khz, 8bit, mono channel, mulaw encoding
RasaAudioBytes = NewType("RasaAudioBytes", bytes)
HERTZ = 8000
