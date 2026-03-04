from dataclasses import dataclass


@dataclass
class StreamingConfig:
    """Configuration for streaming

    Attributes:
        response_text_contains_ssml: Whether the response text contains SSML tags.
    """

    response_text_contains_ssml: bool = False
