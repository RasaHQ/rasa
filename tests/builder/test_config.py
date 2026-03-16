"""Tests for ``rasa.builder.config``."""

import pytest

from rasa.builder import config
from rasa.builder.config import apply_proxy_url


class TestApplyProxyUrl:
    """apply_proxy_url must set PROXY_URL and recompute derived URLs."""

    @pytest.fixture(autouse=True)
    def _snapshot_and_restore(self) -> None:
        """Capture module-level values before each test and restore after."""
        snapshot = {
            attr: getattr(config, attr)
            for attr in (
                "PROXY_URL",
                "INKEEP_BASE_URL",
                "LAKERA_BASE_URL",
                "LANGFUSE_HOST",
                "LANGFUSE_PUBLIC_KEY",
                "LANGFUSE_SECRET_KEY",
            )
        }
        yield
        for attr, value in snapshot.items():
            setattr(config, attr, value)

    def test_sets_proxy_url_and_derived_urls(self) -> None:
        proxy = "https://tools-proxy.example.com"

        apply_proxy_url(proxy)

        assert config.PROXY_URL == proxy
        assert config.INKEEP_BASE_URL == f"{proxy}/documentation"
        assert config.LAKERA_BASE_URL == f"{proxy}/guardrails"
        assert config.LANGFUSE_HOST == f"{proxy}/langfuse"

    def test_strips_trailing_slash(self) -> None:
        proxy = "https://tools-proxy.example.com/"

        apply_proxy_url(proxy)

        assert config.INKEEP_BASE_URL == "https://tools-proxy.example.com/documentation"

    def test_none_clears_proxy_url(self) -> None:
        apply_proxy_url("https://some-proxy.example.com")
        apply_proxy_url(None)

        assert config.PROXY_URL is None

    def test_langfuse_keys_fallback_to_license(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(config, "RASA_PRO_LICENSE", "test-license")
        monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", None)
        monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", None)

        apply_proxy_url("https://tools-proxy.example.com")

        assert config.LANGFUSE_PUBLIC_KEY == "test-license"
        assert config.LANGFUSE_SECRET_KEY == "test-license"
