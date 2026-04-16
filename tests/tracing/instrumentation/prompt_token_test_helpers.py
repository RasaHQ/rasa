"""Helpers shared by LLM prompt-token tracing unit tests."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from pytest import MonkeyPatch

RESOLVE_TIKTOKEN_ENCODE_PATCH_PATH = (
    "rasa.tracing.instrumentation.attribute_extractors.resolve_tiktoken_encode"
)

PROMPT_TOKEN_TEST_LLM_INPUT_TEXT = "This is a test prompt."

NON_OPENAI_ROUTER_MODEL_GROUPS: List[Dict[str, Any]] = [
    {
        "id": "non-openai-router-group",
        "models": [
            {"provider": "cohere", "model": "command", "api_key": "test"},
            {
                "provider": "azure",
                "deployment": "my-azure",
                "api_key": "test",
                "api_base": "test-base",
                "api_version": "test-version",
            },
        ],
        "router": {"routing_strategy": "simple-shuffle"},
    },
]


def patch_resolve_tiktoken_encode_fixed_token_ids(
    monkeypatch: MonkeyPatch,
    token_ids: tuple[int, ...] = (1, 2, 3, 4),
) -> None:
    """Stub tiktoken so prompt length resolves to ``len(token_ids)`` tokens."""

    monkeypatch.setattr(
        RESOLVE_TIKTOKEN_ENCODE_PATCH_PATH,
        lambda model_name, fallback_encoding="cl100k_base": (
            lambda prompt: list(token_ids)
        ),
    )


def patch_mock_contextual_response_rephraser_llm_health_check_noop(
    monkeypatch: MonkeyPatch,
    rephraser_class: Type[Any],
) -> None:
    """No-op LLM health check for mock rephraser tests that use router model groups.

    Patching ``health_check.perform_llm_health_check`` globally breaks other tests in
    the same session (and conflicts with fixtures that patch the same symbol). Setting
    a no-op on the mock rephraser class works with instrumentation whether or not that
    class was already instrumented in a prior test.
    """

    monkeypatch.setattr(
        rephraser_class,
        "perform_llm_health_check",
        staticmethod(lambda *args, **kwargs: None),
    )


def set_tiktoken_cache_dir_env(monkeypatch: MonkeyPatch, directory: str) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", directory)
