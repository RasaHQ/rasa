from __future__ import annotations

from typing import Any, Callable, Sequence

import structlog
import tiktoken

O200K_BASE_ENCODING = "o200k_base"

MODEL_PREFIX_TO_ENCODING_ALIASES = {
    "gpt-5.1-": O200K_BASE_ENCODING,
    "gpt-5.2-": O200K_BASE_ENCODING,
    "gpt-5-mini-": O200K_BASE_ENCODING,
    "gpt-5-nano-": O200K_BASE_ENCODING,
}

MODEL_TO_ENCODING_ALIASES = {
    "gpt-5.1": O200K_BASE_ENCODING,
    "gpt-5.2": O200K_BASE_ENCODING,
    "gpt-5-mini": O200K_BASE_ENCODING,
    "gpt-5-nano": O200K_BASE_ENCODING,
}

structlogger = structlog.get_logger()


def register_tiktoken_model_aliases() -> None:
    """Register runtime model aliases for tiktoken model resolution."""
    import tiktoken.model

    for model_prefix, encoding in MODEL_PREFIX_TO_ENCODING_ALIASES.items():
        tiktoken.model.MODEL_PREFIX_TO_ENCODING.setdefault(model_prefix, encoding)

    for model, encoding in MODEL_TO_ENCODING_ALIASES.items():
        tiktoken.model.MODEL_TO_ENCODING.setdefault(model, encoding)


def resolve_tiktoken_encode(
    model_name: str, fallback_encoding: str = "cl100k_base"
) -> Callable[[str], Sequence[Any]]:
    """Return a resilient encode callable for the given model name."""
    register_tiktoken_model_aliases()

    try:
        return tiktoken.encoding_for_model(model_name).encode
    except KeyError:
        structlogger.warning(
            "tiktoken_utils.resolve_tiktoken_encode.unknown_model",
            event_info=(
                f"Unknown model name '{model_name}', "
                f"using '{fallback_encoding}' encoding as fallback."
            ),
        )
    except Exception as exc:
        structlogger.warning(
            "tiktoken_utils.resolve_tiktoken_encode.error",
            event_info=(
                f"Failed to resolve tokenizer for model '{model_name}', "
                f"using '{fallback_encoding}' encoding as fallback."
            ),
            error=repr(exc),
        )

    try:
        return tiktoken.get_encoding(fallback_encoding).encode
    except Exception:
        return lambda value: value.split()
