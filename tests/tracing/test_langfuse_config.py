import os
import textwrap
from pathlib import Path

import pytest
import structlog

from rasa.tracing.constants import (
    LANGFUSE_CONFIG_BASE_URL_KEY,
    LANGFUSE_CONFIG_DEBUG_KEY,
    LANGFUSE_CONFIG_ENVIRONMENT_KEY,
    LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY,
    LANGFUSE_CONFIG_PRIVATE_KEY,
    LANGFUSE_CONFIG_PUBLIC_KEY,
    LANGFUSE_CONFIG_RELEASE_KEY,
    LANGFUSE_CONFIG_SAMPLE_RATE_KEY,
    LANGFUSE_CONFIG_TIMEOUT_KEY,
)
from rasa.tracing.exceptions import (
    DuplicateTracingConfigException,
    InvalidLangfuseConfigException,
)
from rasa.tracing.langfuse_config import (
    _configure_litellm_callback,
    _extract_langfuse_config_values,
    _get_langfuse_config,
    _log_multiple_langfuse_configs_error,
    _parse_tracing_configs,
    _resolve_environment_variables,
    _set_langfuse_environment_variables,
    _validate_key_syntax,
    _validate_langfuse_config,
    _validate_required_keys,
    configure_langfuse,
)
from tests.utilities import filter_logs


@pytest.mark.parametrize(
    "endpoints_file,expected",
    [
        ("", None),
        (None, None),
    ],
)
def test_get_langfuse_config_empty_file(endpoints_file: str, expected: None) -> None:
    """Test that _get_langfuse_config returns None for empty/None endpoints_file."""
    result = _get_langfuse_config(endpoints_file)
    assert result == expected


def test_get_langfuse_config_file_not_found(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns None when file doesn't exist."""
    non_existent_file = str(tmp_path / "non_existent_endpoints.yml")
    result = _get_langfuse_config(non_existent_file)
    assert result is None


def test_get_langfuse_config_no_tracing_key(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns None when tracing key is missing."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            action_endpoint:
                url: "http://localhost:5056/webhook"
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


def test_get_langfuse_config_no_langfuse_config(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns None when no langfuse config is found."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: jaeger
                host: localhost
                port: 6831
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


def test_get_langfuse_config_single_dict(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns config when langfuse config
    exists as single dict.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${LANGFUSE_PUBLIC_KEY}
                private_key: ${LANGFUSE_SECRET_KEY}
                host: https://cloud.langfuse.com
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is not None
    assert result.type == "langfuse"
    assert result.kwargs[LANGFUSE_CONFIG_PUBLIC_KEY] == "${LANGFUSE_PUBLIC_KEY}"
    assert result.kwargs[LANGFUSE_CONFIG_PRIVATE_KEY] == "${LANGFUSE_SECRET_KEY}"
    assert result.kwargs[LANGFUSE_CONFIG_BASE_URL_KEY] == "https://cloud.langfuse.com"


def test_get_langfuse_config_list_with_langfuse(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns config when langfuse config
    exists in a list.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: jaeger
                  host: localhost
                  port: 6831
                - type: langfuse
                  public_key: ${LANGFUSE_PUBLIC_KEY}
                  private_key: ${LANGFUSE_SECRET_KEY}
                  host: https://cloud.langfuse.com
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is not None
    assert result.type == "langfuse"
    assert result.kwargs[LANGFUSE_CONFIG_PUBLIC_KEY] == "${LANGFUSE_PUBLIC_KEY}"


def test_get_langfuse_config_multiple_langfuse_configs(tmp_path: Path) -> None:
    """Test that _get_langfuse_config raises exception when multiple langfuse
    configs found.
    """
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                - type: langfuse
                  public_key: ${LANGFUSE_PUBLIC_KEY_1}
                  private_key: ${LANGFUSE_SECRET_KEY_1}
                  host: https://cloud.langfuse.com
                - type: langfuse
                  public_key: ${LANGFUSE_PUBLIC_KEY_2}
                  private_key: ${LANGFUSE_SECRET_KEY_2}
                  host: https://cloud.langfuse.com
            """
        )
    )
    with pytest.raises(DuplicateTracingConfigException) as exc_info:
        _get_langfuse_config(str(endpoints_file))
    assert "Multiple Langfuse configs found" in str(exc_info.value)


def test_get_langfuse_config_handles_exception(tmp_path: Path, monkeypatch) -> None:
    """Test that _get_langfuse_config handles exceptions gracefully."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${LANGFUSE_PUBLIC_KEY}
                private_key: ${LANGFUSE_SECRET_KEY}
                host: https://cloud.langfuse.com
            """
        )
    )

    # Mock read_config_file to raise an exception
    def mock_read_config_file(filename: str) -> None:
        raise FileNotFoundError("Test exception")

    monkeypatch.setattr(
        "rasa.tracing.langfuse_config.read_config_file", mock_read_config_file
    )

    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


@pytest.mark.parametrize(
    "config_values,expected_exception,expected_message",
    [
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "",
                LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
                LANGFUSE_CONFIG_PRIVATE_KEY: "",
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
                LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
                LANGFUSE_CONFIG_BASE_URL_KEY: "",
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: None,
                LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
                LANGFUSE_CONFIG_PRIVATE_KEY: None,
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
                LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
                LANGFUSE_CONFIG_BASE_URL_KEY: None,
            },
            InvalidLangfuseConfigException,
            "required",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "public_key",
                LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "${syntax}",
        ),
        (
            {
                LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
                LANGFUSE_CONFIG_PRIVATE_KEY: "secret_key",
                LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
            },
            InvalidLangfuseConfigException,
            "${syntax}",
        ),
    ],
)
def test_validate_langfuse_config_invalid(
    config_values: dict,
    expected_exception: type,
    expected_message: str,
) -> None:
    """Test that _validate_langfuse_config raises exception for invalid configs."""
    with pytest.raises(expected_exception) as exc_info:
        _validate_langfuse_config(config_values)
    assert expected_message.lower() in str(exc_info.value).lower()


def test_validate_langfuse_config_valid() -> None:
    """Test that _validate_langfuse_config passes for valid config."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "${LANGFUSE_PUBLIC_KEY}",
        LANGFUSE_CONFIG_PRIVATE_KEY: "${LANGFUSE_SECRET_KEY}",
        LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
    }
    # Should not raise any exception
    _validate_langfuse_config(config_values)


def test_configure_langfuse_no_config(tmp_path: Path) -> None:
    """Test that configure_langfuse returns early when no config found."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            action_endpoint:
                url: "http://localhost:5056/webhook"
            """
        )
    )

    with structlog.testing.capture_logs() as caplog:
        configure_langfuse(str(endpoints_file))
        logs = filter_logs(caplog, "langfuse_configuration.langfuse_config_not_found")

        assert logs is not None


def test_configure_langfuse_sets_environment_variables(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that configure_langfuse sets environment variables correctly."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: "30"
                debug: "true"
                environment: production
                release: v1.0.0
                media_upload_thread_count: "5"
                sample_rate: "0.5"
            """
        )
    )

    # Set environment variables
    monkeypatch.setenv("TEST_PUBLIC_KEY", "resolved_public_key")
    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret_key")

    # Clear any existing langfuse env vars
    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
        "LANGFUSE_TIMEOUT",
        "LANGFUSE_DEBUG",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "LANGFUSE_RELEASE",
        "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT",
        "LANGFUSE_SAMPLE_RATE",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    configure_langfuse(str(endpoints_file))

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "resolved_public_key"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "resolved_secret_key"
    assert os.environ["LANGFUSE_OTEL_HOST"] == "https://cloud.langfuse.com"
    assert os.environ["LANGFUSE_TIMEOUT"] == "30"
    assert os.environ["LANGFUSE_DEBUG"] == "true"
    assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "production"
    assert os.environ["LANGFUSE_RELEASE"] == "v1.0.0"
    assert os.environ["LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT"] == "5"
    assert os.environ["LANGFUSE_SAMPLE_RATE"] == "0.5"


def test_configure_langfuse_minimal_config(tmp_path: Path, monkeypatch) -> None:
    """Test that configure_langfuse works with minimal required config."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: null
                debug: null
                environment: null
                release: null
                media_upload_thread_count: null
                sample_rate: null
            """
        )
    )

    monkeypatch.setenv("TEST_PUBLIC_KEY", "resolved_public_key")
    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret_key")

    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    configure_langfuse(str(endpoints_file))

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "resolved_public_key"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "resolved_secret_key"
    assert os.environ["LANGFUSE_OTEL_HOST"] == "https://cloud.langfuse.com"


def test_configure_langfuse_invalid_config_raises_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that configure_langfuse raises exception for invalid config."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: invalid_no_dollar_sign
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: null
                debug: null
                environment: null
                release: null
                media_upload_thread_count: null
                sample_rate: null
            """
        )
    )

    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret_key")

    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        configure_langfuse(str(endpoints_file))
    assert (
        "${syntax}" in str(exc_info.value).lower()
        or "public_key" in str(exc_info.value).lower()
    )


def test_configure_langfuse_sets_litellm_callback(tmp_path: Path, monkeypatch) -> None:
    """Test that configure_langfuse sets litellm.success_callback."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: null
                debug: null
                environment: null
                release: null
                media_upload_thread_count: null
                sample_rate: null
            """
        )
    )

    monkeypatch.setenv("TEST_PUBLIC_KEY", "resolved_public_key")
    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret_key")

    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    configure_langfuse(str(endpoints_file))


def test_configure_langfuse_optional_parameters_none(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that configure_langfuse handles None optional parameters correctly."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: null
                debug: null
                environment: null
                release: null
                media_upload_thread_count: null
                sample_rate: null
            """
        )
    )

    monkeypatch.setenv("TEST_PUBLIC_KEY", "resolved_public_key")
    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret_key")

    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
        "LANGFUSE_TIMEOUT",
        "LANGFUSE_DEBUG",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "LANGFUSE_RELEASE",
        "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT",
        "LANGFUSE_SAMPLE_RATE",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    configure_langfuse(str(endpoints_file))

    # Required params should be set
    assert "LANGFUSE_PUBLIC_KEY" in os.environ
    assert "LANGFUSE_SECRET_KEY" in os.environ
    assert "LANGFUSE_OTEL_HOST" in os.environ

    # Optional params should not be in env if they were None in config
    assert "LANGFUSE_TIMEOUT" not in os.environ
    assert "LANGFUSE_DEBUG" not in os.environ
    assert "LANGFUSE_TRACING_ENVIRONMENT" not in os.environ
    assert "LANGFUSE_RELEASE" not in os.environ
    assert "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT" not in os.environ
    assert "LANGFUSE_SAMPLE_RATE" not in os.environ


# ------------------------------------------------------------
# Helper function tests
# ------------------------------------------------------------


def test_extract_langfuse_config_values(tmp_path: Path) -> None:
    """Test that _extract_langfuse_config_values extracts all config values."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: "30"
                debug: "true"
            """
        )
    )

    langfuse_config = _get_langfuse_config(str(endpoints_file))
    assert langfuse_config is not None

    config_values = _extract_langfuse_config_values(langfuse_config)

    assert config_values[LANGFUSE_CONFIG_PUBLIC_KEY] == "${TEST_PUBLIC_KEY}"
    assert config_values[LANGFUSE_CONFIG_PRIVATE_KEY] == "${TEST_SECRET_KEY}"
    assert config_values[LANGFUSE_CONFIG_BASE_URL_KEY] == "https://cloud.langfuse.com"
    assert config_values[LANGFUSE_CONFIG_TIMEOUT_KEY] == "30"
    assert config_values[LANGFUSE_CONFIG_DEBUG_KEY] == "true"


def test_resolve_environment_variables(monkeypatch) -> None:
    """Test that _resolve_environment_variables resolves env vars correctly."""
    monkeypatch.setenv("TEST_PUBLIC_KEY", "resolved_public")
    monkeypatch.setenv("TEST_SECRET_KEY", "resolved_secret")

    resolved = _resolve_environment_variables(
        "${TEST_PUBLIC_KEY}", "${TEST_SECRET_KEY}"
    )

    assert resolved[LANGFUSE_CONFIG_PUBLIC_KEY] == "resolved_public"
    assert resolved[LANGFUSE_CONFIG_PRIVATE_KEY] == "resolved_secret"


def test_get_langfuse_config_tracing_config_is_none(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns None when tracing config is None."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing: null
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


def test_get_langfuse_config_tracing_config_empty_list(tmp_path: Path) -> None:
    """Test that _get_langfuse_config returns None when tracing config is empty list."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing: []
            """
        )
    )
    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


def test_get_langfuse_config_handles_generic_exception(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that _get_langfuse_config handles generic exceptions gracefully."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
            """
        )
    )

    # Mock read_config_file to raise a generic exception
    def mock_read_config_file(filename: str) -> None:
        raise ValueError("Test generic exception")

    monkeypatch.setattr(
        "rasa.tracing.langfuse_config.read_config_file", mock_read_config_file
    )

    result = _get_langfuse_config(str(endpoints_file))
    assert result is None


def test_parse_tracing_configs_with_dict() -> None:
    """Test that _parse_tracing_configs handles dict config correctly."""
    config = {
        "type": "langfuse",
        "public_key": "${TEST_PUBLIC_KEY}",
        "private_key": "${TEST_SECRET_KEY}",
    }
    result = _parse_tracing_configs(config)
    assert len(result) == 1
    assert result[0].type == "langfuse"


def test_parse_tracing_configs_with_list() -> None:
    """Test that _parse_tracing_configs handles list config correctly."""
    config = [
        {"type": "jaeger", "host": "localhost"},
        {"type": "langfuse", "public_key": "${TEST_PUBLIC_KEY}"},
    ]
    result = _parse_tracing_configs(config)
    assert len(result) == 2
    assert result[0].type == "jaeger"
    assert result[1].type == "langfuse"


def test_log_multiple_langfuse_configs_error(tmp_path: Path) -> None:
    """Test that _log_multiple_langfuse_configs_error logs error correctly."""
    endpoints_file = tmp_path / "endpoints.yml"
    with structlog.testing.capture_logs() as caplog:
        _log_multiple_langfuse_configs_error(str(endpoints_file))
        logs = filter_logs(caplog, "langfuse_configuration.multiple_langfuse_configs")
        assert logs is not None
        assert "Multiple Langfuse configs found" in logs[0]["event_info"]


def test_set_langfuse_environment_variables_all_values(monkeypatch) -> None:
    """Test that _set_langfuse_environment_variables sets all environment variables."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "test_public",
        LANGFUSE_CONFIG_PRIVATE_KEY: "test_secret",
        LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
        LANGFUSE_CONFIG_TIMEOUT_KEY: "30",
        LANGFUSE_CONFIG_DEBUG_KEY: "true",
        LANGFUSE_CONFIG_ENVIRONMENT_KEY: "production",
        LANGFUSE_CONFIG_RELEASE_KEY: "v1.0.0",
        LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY: "5",
        LANGFUSE_CONFIG_SAMPLE_RATE_KEY: "0.5",
    }

    resolved_keys = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "resolved_public",
        LANGFUSE_CONFIG_PRIVATE_KEY: "resolved_secret",
    }

    # Clear environment variables
    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
        "LANGFUSE_TIMEOUT",
        "LANGFUSE_DEBUG",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "LANGFUSE_RELEASE",
        "LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT",
        "LANGFUSE_SAMPLE_RATE",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    _set_langfuse_environment_variables(config_values, resolved_keys)

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "resolved_public"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "resolved_secret"
    assert os.environ["LANGFUSE_OTEL_HOST"] == "https://cloud.langfuse.com"
    assert os.environ["LANGFUSE_TIMEOUT"] == "30"
    assert os.environ["LANGFUSE_DEBUG"] == "true"
    assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "production"
    assert os.environ["LANGFUSE_RELEASE"] == "v1.0.0"
    assert os.environ["LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT"] == "5"
    assert os.environ["LANGFUSE_SAMPLE_RATE"] == "0.5"


def test_set_langfuse_environment_variables_none_values(monkeypatch) -> None:
    """Test that _set_langfuse_environment_variables skips None values."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: None,
        LANGFUSE_CONFIG_PRIVATE_KEY: None,
        LANGFUSE_CONFIG_BASE_URL_KEY: None,
        LANGFUSE_CONFIG_TIMEOUT_KEY: None,
        LANGFUSE_CONFIG_DEBUG_KEY: None,
        LANGFUSE_CONFIG_ENVIRONMENT_KEY: None,
        LANGFUSE_CONFIG_RELEASE_KEY: None,
        LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY: None,
        LANGFUSE_CONFIG_SAMPLE_RATE_KEY: None,
    }

    resolved_keys = {
        LANGFUSE_CONFIG_PUBLIC_KEY: None,
        LANGFUSE_CONFIG_PRIVATE_KEY: None,
    }

    # Clear environment variables
    env_vars_to_clear = [
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_OTEL_HOST",
        "LANGFUSE_TIMEOUT",
        "LANGFUSE_DEBUG",
    ]
    for var in env_vars_to_clear:
        monkeypatch.delenv(var, raising=False)

    _set_langfuse_environment_variables(config_values, resolved_keys)

    # None values should not be set
    assert "LANGFUSE_PUBLIC_KEY" not in os.environ
    assert "LANGFUSE_SECRET_KEY" not in os.environ
    assert "LANGFUSE_OTEL_HOST" not in os.environ
    assert "LANGFUSE_TIMEOUT" not in os.environ
    assert "LANGFUSE_DEBUG" not in os.environ


def test_configure_litellm_callback() -> None:
    """Test that _configure_litellm_callback sets litellm.success_callback."""
    # Import litellm here to avoid import errors if not installed
    try:
        import litellm

        # Save original value
        original_callback = getattr(litellm, "success_callback", None)

        _configure_litellm_callback()

        assert litellm.success_callback == ["langfuse_otel"]

        # Restore original value
        if original_callback is not None:
            litellm.success_callback = original_callback
    except ImportError:
        pytest.skip("litellm not installed")


def test_validate_required_keys_missing_keys() -> None:
    """Test that _validate_required_keys raises exception when keys are missing."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "",
        LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
        LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
    }

    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_required_keys(
            config_values,
            [
                LANGFUSE_CONFIG_PUBLIC_KEY,
                LANGFUSE_CONFIG_PRIVATE_KEY,
                LANGFUSE_CONFIG_BASE_URL_KEY,
            ],
        )
    assert "required" in str(exc_info.value).lower()
    assert LANGFUSE_CONFIG_PUBLIC_KEY in str(exc_info.value)


def test_validate_required_keys_missing_multiple_keys() -> None:
    """Test that _validate_required_keys raises exception with multiple missing keys."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "",
        LANGFUSE_CONFIG_PRIVATE_KEY: None,
        LANGFUSE_CONFIG_BASE_URL_KEY: "",
    }

    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_required_keys(
            config_values,
            [
                LANGFUSE_CONFIG_PUBLIC_KEY,
                LANGFUSE_CONFIG_PRIVATE_KEY,
                LANGFUSE_CONFIG_BASE_URL_KEY,
            ],
        )
    assert "required" in str(exc_info.value).lower()
    # Should mention all missing keys
    error_message = str(exc_info.value).lower()
    assert LANGFUSE_CONFIG_PUBLIC_KEY in error_message or "public_key" in error_message


def test_validate_required_keys_all_present() -> None:
    """Test that _validate_required_keys passes when all keys are present."""
    config_values = {
        LANGFUSE_CONFIG_PUBLIC_KEY: "${PUBLIC}",
        LANGFUSE_CONFIG_PRIVATE_KEY: "${SECRET}",
        LANGFUSE_CONFIG_BASE_URL_KEY: "https://cloud.langfuse.com",
    }

    # Should not raise any exception
    _validate_required_keys(
        config_values,
        [
            LANGFUSE_CONFIG_PUBLIC_KEY,
            LANGFUSE_CONFIG_PRIVATE_KEY,
            LANGFUSE_CONFIG_BASE_URL_KEY,
        ],
    )


def test_validate_key_syntax_invalid_public_key() -> None:
    """Test that _validate_key_syntax raises exception for invalid public key."""
    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("invalid_no_dollar", "${SECRET}")
    assert "${syntax}" in str(exc_info.value).lower()
    assert "public_key" in str(exc_info.value).lower()


def test_validate_key_syntax_invalid_secret_key() -> None:
    """Test that _validate_key_syntax raises exception for invalid secret key."""
    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("${PUBLIC}", "invalid_no_dollar")
    assert "${syntax}" in str(exc_info.value).lower()
    assert "private_key" in str(exc_info.value).lower()


def test_validate_key_syntax_both_invalid() -> None:
    """Test that _validate_key_syntax raises exception when both keys are invalid."""
    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("invalid_public", "invalid_secret")
    assert "${syntax}" in str(exc_info.value).lower()


def test_validate_key_syntax_rejects_dollar_without_braces() -> None:
    """Test that _validate_key_syntax rejects $VAR syntax (only ${VAR} is accepted)."""
    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("$PUBLIC_KEY", "${SECRET_KEY}")
    assert "${syntax}" in str(exc_info.value).lower()

    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("${PUBLIC_KEY}", "$SECRET_KEY")
    assert "${syntax}" in str(exc_info.value).lower()

    with pytest.raises(InvalidLangfuseConfigException) as exc_info:
        _validate_key_syntax("$PUBLIC_KEY", "$SECRET_KEY")
    assert "${syntax}" in str(exc_info.value).lower()


def test_validate_key_syntax_valid() -> None:
    """Test that _validate_key_syntax passes for valid ${VAR} syntax."""
    # Should not raise any exception
    _validate_key_syntax("${PUBLIC_KEY}", "${SECRET_KEY}")


def test_extract_langfuse_config_values_all_fields(tmp_path: Path) -> None:
    """Test that _extract_langfuse_config_values extracts all fields correctly."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
                timeout: "30"
                debug: "true"
                environment: production
                release: v1.0.0
                media_upload_thread_count: "5"
                sample_rate: "0.5"
            """
        )
    )

    langfuse_config = _get_langfuse_config(str(endpoints_file))
    assert langfuse_config is not None

    config_values = _extract_langfuse_config_values(langfuse_config)

    assert config_values[LANGFUSE_CONFIG_PUBLIC_KEY] == "${TEST_PUBLIC_KEY}"
    assert config_values[LANGFUSE_CONFIG_PRIVATE_KEY] == "${TEST_SECRET_KEY}"
    assert config_values[LANGFUSE_CONFIG_BASE_URL_KEY] == "https://cloud.langfuse.com"
    assert config_values[LANGFUSE_CONFIG_TIMEOUT_KEY] == "30"
    assert config_values[LANGFUSE_CONFIG_DEBUG_KEY] == "true"
    assert config_values[LANGFUSE_CONFIG_ENVIRONMENT_KEY] == "production"
    assert config_values[LANGFUSE_CONFIG_RELEASE_KEY] == "v1.0.0"
    assert config_values[LANGFUSE_CONFIG_MEDIA_UPLOAD_THREAD_COUNT_KEY] == "5"
    assert config_values[LANGFUSE_CONFIG_SAMPLE_RATE_KEY] == "0.5"


def test_langfuse_keys_not_expanded_during_yaml_parsing(
    tmp_path: Path, monkeypatch
) -> None:
    """Test Langfuse public_key and private_key are not expanded during YAML parsing."""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY_TEST", "expanded_public_key_value")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_TEST", "expanded_secret_key_value")

    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${LANGFUSE_PUBLIC_KEY_TEST}
                private_key: ${LANGFUSE_SECRET_KEY_TEST}
                host: https://cloud.langfuse.com
            """
        )
    )

    langfuse_config = _get_langfuse_config(str(endpoints_file))
    assert langfuse_config is not None

    # The keys should NOT be expanded - they should still contain ${VAR} syntax
    assert (
        langfuse_config.kwargs[LANGFUSE_CONFIG_PUBLIC_KEY]
        == "${LANGFUSE_PUBLIC_KEY_TEST}"
    )
    assert (
        langfuse_config.kwargs[LANGFUSE_CONFIG_PRIVATE_KEY]
        == "${LANGFUSE_SECRET_KEY_TEST}"
    )

    # Verify they are NOT the expanded values
    assert (
        langfuse_config.kwargs[LANGFUSE_CONFIG_PUBLIC_KEY]
        != "expanded_public_key_value"
    )
    assert (
        langfuse_config.kwargs[LANGFUSE_CONFIG_PRIVATE_KEY]
        != "expanded_secret_key_value"
    )


def test_extract_langfuse_config_values_missing_optional_fields(
    tmp_path: Path,
) -> None:
    """Test that _extract_langfuse_config_values handles missing optional fields."""
    endpoints_file = tmp_path / "endpoints.yml"
    endpoints_file.write_text(
        textwrap.dedent(
            """
            tracing:
                type: langfuse
                public_key: ${TEST_PUBLIC_KEY}
                private_key: ${TEST_SECRET_KEY}
                host: https://cloud.langfuse.com
            """
        )
    )

    langfuse_config = _get_langfuse_config(str(endpoints_file))
    assert langfuse_config is not None

    config_values = _extract_langfuse_config_values(langfuse_config)

    assert config_values[LANGFUSE_CONFIG_PUBLIC_KEY] == "${TEST_PUBLIC_KEY}"
    assert config_values[LANGFUSE_CONFIG_PRIVATE_KEY] == "${TEST_SECRET_KEY}"
    assert config_values[LANGFUSE_CONFIG_BASE_URL_KEY] == "https://cloud.langfuse.com"
    # Optional fields should be None
    assert config_values[LANGFUSE_CONFIG_TIMEOUT_KEY] is None
    assert config_values[LANGFUSE_CONFIG_DEBUG_KEY] is None
