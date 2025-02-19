import argparse
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
from pytest import RunResult

from rasa.cli.inspect import inspect
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
