import re

_SSML_TAG_PATTERN = re.compile(
    r"<\s*/?\s*(prosody|phoneme|break|emphasis|say-as|sub|mark|"
    r"s|p|lang|audio|bookmark)\b",
    re.IGNORECASE,
)


def contains_ssml_tags(text: str) -> bool:
    """Check if text contains SSML markup tags.

    Excludes <speak> and <voice> since those are always added by
    the TTS engine's request body wrapper.
    """
    return bool(_SSML_TAG_PATTERN.search(text))
