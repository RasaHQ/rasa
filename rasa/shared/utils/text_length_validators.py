from typing import Text, Dict, Any


def validate_text_length_in_characters(text: Text, limit: int) -> bool:
    """Return whether the given text exceeds the set character limit."""
    return len(text) > limit


def validate_text_length_in_words(text: Text, limit: int) -> bool:
    """Return whether the given text exceeds the set word limit."""
    return len(text.strip().split()) > limit


def validate_text_length_in_tokens(
    text: Text, limit: int, config: Dict[Text, Any]
) -> bool:
    # TODO: Implement tokenizer factory and tokenizer interface before this
    raise NotImplementedError
