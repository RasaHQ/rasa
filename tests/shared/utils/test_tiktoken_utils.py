from unittest.mock import Mock

import tiktoken

from rasa.shared.utils.tiktoken_utils import (
    O200K_BASE_ENCODING,
    register_tiktoken_model_aliases,
    resolve_tiktoken_encode,
)


def test_register_tiktoken_model_aliases_adds_gpt_5_family_mappings() -> None:
    import tiktoken.model

    model_prefixes = [
        "gpt-5.1-",
        "gpt-5.2-",
        "gpt-5-mini-",
        "gpt-5-nano-",
    ]
    models = ["gpt-5.1", "gpt-5.2", "gpt-5-mini", "gpt-5-nano"]

    previous_prefix_values = {
        model_prefix: tiktoken.model.MODEL_PREFIX_TO_ENCODING.get(model_prefix)
        for model_prefix in model_prefixes
    }
    previous_model_values = {
        model: tiktoken.model.MODEL_TO_ENCODING.get(model) for model in models
    }

    try:
        for model_prefix in model_prefixes:
            tiktoken.model.MODEL_PREFIX_TO_ENCODING.pop(model_prefix, None)
        for model in models:
            tiktoken.model.MODEL_TO_ENCODING.pop(model, None)

        register_tiktoken_model_aliases()

        for model_prefix in model_prefixes:
            assert (
                tiktoken.model.MODEL_PREFIX_TO_ENCODING.get(model_prefix)
                == O200K_BASE_ENCODING
            )
        for model in models:
            assert tiktoken.model.MODEL_TO_ENCODING.get(model) == O200K_BASE_ENCODING
    finally:
        for model_prefix in model_prefixes:
            if previous_prefix_values[model_prefix] is None:
                tiktoken.model.MODEL_PREFIX_TO_ENCODING.pop(model_prefix, None)
            else:
                tiktoken.model.MODEL_PREFIX_TO_ENCODING[model_prefix] = (
                    previous_prefix_values[model_prefix]
                )

        for model in models:
            if previous_model_values[model] is None:
                tiktoken.model.MODEL_TO_ENCODING.pop(model, None)
            else:
                tiktoken.model.MODEL_TO_ENCODING[model] = previous_model_values[model]


def test_resolve_tiktoken_encode_uses_model_encoding(monkeypatch) -> None:
    fake_encoding = Mock()
    fake_encoding.encode.return_value = [1, 2]

    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda _: fake_encoding)
    monkeypatch.setattr(
        tiktoken,
        "get_encoding",
        lambda _: (_ for _ in ()).throw(AssertionError("fallback not expected")),
    )

    encode = resolve_tiktoken_encode("gpt-5.1-2025-11-13")

    assert encode("hello") == [1, 2]


def test_resolve_tiktoken_encode_uses_fallback_encoding(monkeypatch) -> None:
    fake_fallback_encoding = Mock()
    fake_fallback_encoding.encode.return_value = [1]

    def _raise_key_error(_: str):
        raise KeyError("unknown model")

    monkeypatch.setattr(tiktoken, "encoding_for_model", _raise_key_error)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _: fake_fallback_encoding)

    encode = resolve_tiktoken_encode("some-unknown-model")

    assert encode("hello") == [1]


def test_resolve_tiktoken_encode_uses_split_fallback_when_tiktoken_unavailable(
    monkeypatch,
) -> None:
    def _raise_runtime_error(_: str):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(tiktoken, "encoding_for_model", _raise_runtime_error)
    monkeypatch.setattr(tiktoken, "get_encoding", _raise_runtime_error)

    encode = resolve_tiktoken_encode("gpt-5.1-2025-11-13")

    assert encode("a bb ccc") == ["a", "bb", "ccc"]
