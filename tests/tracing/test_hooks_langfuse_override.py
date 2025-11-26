from typing import Optional

import pytest

from rasa.__main__ import create_argument_parser
from rasa.hooks import configure_commandline


def test_configure_commandline_passes_e2e_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When executing E2E tests, langfuse_environment_name should be 'e2e-test'."""
    # Capture the override passed into configure_langfuse
    captured: dict[str, Optional[str]] = {}

    def fake_configure_langfuse(
        endpoints_file: str, langfuse_environment_name: Optional[str] = None
    ) -> None:
        captured["langfuse_environment_name"] = langfuse_environment_name

    monkeypatch.setattr(
        "rasa.tracing.langfuse_config.configure_langfuse", fake_configure_langfuse
    )

    monkeypatch.setattr(
        "rasa.cli.x._get_credentials_and_endpoints_paths",
        lambda args: (None, "dummy_endpoints_path"),
    )

    parser = create_argument_parser()
    args = parser.parse_args(["test", "e2e"])

    configure_commandline(args)  # type: ignore[arg-type]

    assert captured.get("langfuse_environment_name") == "e2e-test"


def test_configure_commandline_passes_du_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When executing DU tests, langfuse_environment_name should be 'du-test'."""
    captured: dict[str, Optional[str]] = {}

    def fake_configure_langfuse(
        endpoints_file: str, langfuse_environment_name: Optional[str] = None
    ) -> None:
        captured["langfuse_environment_name"] = langfuse_environment_name

    monkeypatch.setattr(
        "rasa.tracing.langfuse_config.configure_langfuse", fake_configure_langfuse
    )

    parser = create_argument_parser()
    args = parser.parse_args(["test", "du"])

    configure_commandline(args)  # type: ignore[arg-type]

    assert captured.get("langfuse_environment_name") == "du-test"
