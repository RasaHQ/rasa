import argparse
import asyncio
from pathlib import Path
from typing import Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pytest import RunResult

from rasa.cli.inspect import inspect
from rasa.core import constants
from rasa.core.config.credentials import CredentialsConfig
from rasa.shared.core.domain import Domain

run_module_path = "rasa.cli.run"


@pytest.fixture
def mock_rasa_run(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Mocks the `rasa.cli.run.run` function."""
    _mock_rasa_run = MagicMock()

    monkeypatch.setattr(f"{run_module_path}.rasa_run", _mock_rasa_run)
    return _mock_rasa_run


def test_rasa_inspect_help(run: Callable[..., RunResult]) -> None:
    help_text = """usage: rasa inspect [-h] [-v] [-vv] [--quiet]
                    [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                    [--log-file LOG_FILE] [--use-syslog]
                    [--syslog-address SYSLOG_ADDRESS]
                    [--syslog-port SYSLOG_PORT]
                    [--syslog-protocol SYSLOG_PROTOCOL]
                    [--endpoints ENDPOINTS] [-i INTERFACE] [-p PORT]
                    [--sub-agents SUB_AGENTS]
                    [--response-timeout RESPONSE_TIMEOUT]
                    [--request-timeout REQUEST_TIMEOUT]
                    [--remote-storage REMOTE_STORAGE]
                    [--ssl-certificate SSL_CERTIFICATE]
                    [--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE]
                    [--ssl-password SSL_PASSWORD] [--jwt-secret JWT_SECRET]
                    [--jwt-method JWT_METHOD]
                    [--jwt-private-key JWT_PRIVATE_KEY]
                    [model-as-positional-argument]"""
    lines = help_text.split("\n")

    output = run("inspect", "--help")
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_inspect_invokes_cli_run_with_local_model(
    inspect_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests whether the `rasa inspect` command invokes `rasa run` with a local model."""  # noqa: E501
    # Parse the arguments with which Rasa inspect will be run
    args = inspect_parser.parse_args(
        [
            "inspect",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
        ]
    )

    # Run the inspect function
    inspect(args)

    # Assert that the arguments are correctly passed to the `rasa run` command
    assert args.model == f"{trained_simple_project}/models"
    assert args.endpoints == f"{trained_simple_project}/endpoints.yml"
    assert args.connector == "socketio"

    mock_rasa_run.assert_called_once_with(**vars(args))


def test_cli_run_with_skip_yaml_validation(
    inspect_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests whether the `rasa inspect` command is invoked to skip Domain YAML validation."""  # noqa: E501
    # Parse the arguments with which Rasa inspect will be run
    args = inspect_parser.parse_args(
        [
            "inspect",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
            "--skip-yaml-validation",
            "domain",
        ]
    )

    # Assert that default value for `Domain.validate_yaml` is True
    assert Domain.validate_yaml

    # Run the inspect function
    inspect(args)

    # Assert that the arguments are correctly passed to the `rasa run` command
    assert args.model == f"{trained_simple_project}/models"
    assert args.endpoints == f"{trained_simple_project}/endpoints.yml"
    assert args.skip_yaml_validation == ["domain"]
    assert not Domain.validate_yaml

    mock_rasa_run.assert_called_once_with(**vars(args))


def test_inspect_nextgen_sets_inspector_connector(
    inspect_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
) -> None:
    """Tests whether `rasa inspect --nextgen` uses nextgen inspector channel."""
    args = inspect_parser.parse_args(
        [
            "inspect",
            "--nextgen",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
        ]
    )

    inspect(args)

    assert args.connector == "inspector"
    mock_rasa_run.assert_called_once_with(**vars(args))


def test_inspect_uses_server_url_from_credentials(
    inspect_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
) -> None:
    """Tests that open_inspector_in_browser receives server_url from credentials."""
    custom_server_url = "https://my-rasa-server.example.com"

    mock_credentials = MagicMock(spec=CredentialsConfig)
    mock_credentials.channels = {"inspector": {"server_url": custom_server_url}}

    args = inspect_parser.parse_args(
        [
            "inspect",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
        ]
    )

    with (
        patch(
            "rasa.cli.inspect.CredentialsConfigPath.validate",
            return_value=Path("credentials.yml"),
        ),
        patch(
            "rasa.cli.inspect.CredentialsConfig.load_from_file",
            return_value=mock_credentials,
        ),
        patch(
            "rasa.cli.inspect.open_inspector_in_browser",
            new_callable=AsyncMock,
        ) as mock_open,
    ):
        inspect(args)
        hook, _ = args.server_listeners[0]
        asyncio.run(hook(None, None))

    mock_open.assert_called_once_with(
        custom_server_url, args.voice, args.nextgen, args.auth_token
    )


def test_inspect_falls_back_to_default_server_url_when_no_credentials(
    inspect_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
) -> None:
    """Tests that open_inspector_in_browser falls back to default URL with no credentials."""  # noqa: E501
    args = inspect_parser.parse_args(
        [
            "inspect",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
        ]
    )

    with (
        patch("rasa.cli.inspect.CredentialsConfigPath.validate", return_value=None),
        patch(
            "rasa.cli.inspect.open_inspector_in_browser",
            new_callable=AsyncMock,
        ) as mock_open,
    ):
        inspect(args)
        hook, _ = args.server_listeners[0]
        asyncio.run(hook(None, None))

    expected_url = constants.DEFAULT_SERVER_FORMAT.format("http", args.port)
    mock_open.assert_called_once_with(
        expected_url, args.voice, args.nextgen, args.auth_token
    )
