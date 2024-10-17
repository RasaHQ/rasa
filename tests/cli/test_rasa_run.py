import argparse
import os
import sys
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
from _pytest.pytester import RunResult

from rasa.cli.run import run as cli_run
from rasa.core.persistor import RemoteStorageType
from rasa.shared.core.domain import Domain
from tests.cli.conftest import RASA_EXE

run_module_path = "rasa.cli.run"


@pytest.fixture
def mock_rasa_run(monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Mocks the `rasa.cli.run.run` function."""
    _mock_rasa_run = MagicMock()

    monkeypatch.setattr(f"{run_module_path}.rasa_run", _mock_rasa_run)
    return _mock_rasa_run


def test_run_does_not_start(run_in_simple_project: Callable[..., RunResult]):
    os.remove("domain.yml")

    # the server should not start as no model is configured
    output = run_in_simple_project("run")

    error = "No model found. You have three options to provide a model:"

    assert any(error in line for line in output.outlines)


def test_run_help(
    run: Callable[..., RunResult],
):
    output = run("run", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN] [--cors [CORS ...]]
                [--enable-api] [--response-timeout RESPONSE_TIMEOUT]"""  # noqa: E501
    else:
        version_dependent = """[-i INTERFACE] [-p PORT] [-t AUTH_TOKEN]
                [--cors [CORS [CORS ...]]] [--enable-api]
                [--response-timeout RESPONSE_TIMEOUT]"""

    help_text = (
        f"""usage: {RASA_EXE} run [-h] [-v] [-vv] [--quiet]
                [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL]
                [--log-file LOG_FILE] [--use-syslog]
                [--syslog-address SYSLOG_ADDRESS] [--syslog-port SYSLOG_PORT]
                [--syslog-protocol SYSLOG_PROTOCOL] [--endpoints ENDPOINTS]
                """
        + version_dependent
        + """
                [--remote-storage REMOTE_STORAGE]
                [--ssl-certificate SSL_CERTIFICATE]
                [--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE]
                [--ssl-password SSL_PASSWORD] [--credentials CREDENTIALS]
                [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
                [--jwt-method JWT_METHOD] [--jwt-private-key JWT_PRIVATE_KEY]
                {actions} ... [model-as-positional-argument]"""
    )

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_run_action_help(
    run: Callable[..., RunResult],
):
    output = run("run", "actions", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        help_text = f"""usage: {RASA_EXE} run actions [-h] [-v] [-vv] [--quiet]
                        [--logging-config-file LOGGING_CONFIG_FILE] [-p PORT]
                        [--cors [CORS ...]] [--actions ACTIONS]"""
    else:
        help_text = f"""usage: {RASA_EXE} run actions [-h] [-v] [-vv] [--quiet]
                        [--logging-config-file LOGGING_CONFIG_FILE] [-p PORT]
                        [--cors [CORS [CORS ...]]] [--actions ACTIONS]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_cli_run_with_local_model(
    run_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests whether the `rasa run` command is invoked with a local model."""
    # Parse the arguments with which Rasa inspect will be run
    args = run_parser.parse_args(
        [
            "run",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            f"{trained_simple_project}/models",
        ]
    )

    # Run the rasa.cli.run.run function
    cli_run(args)

    # Assert that the arguments are correctly passed to the `rasa run` command
    assert args.model == f"{trained_simple_project}/models"
    assert args.endpoints == f"{trained_simple_project}/endpoints.yml"

    mock_rasa_run.assert_called_once_with(**vars(args))


@pytest.mark.parametrize(
    "remote_storage, expected_remote_storage",
    [
        ("aws", RemoteStorageType.AWS),
        ("gcs", RemoteStorageType.GCS),
        ("azure", RemoteStorageType.AZURE),
    ],
)
def test_cli_run_with_remote_storage(
    remote_storage: str,
    expected_remote_storage: RemoteStorageType,
    run_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests whether the `rasa run` command is invoked with remote storage."""
    # Parse the arguments with which Rasa will be run
    args = run_parser.parse_args(
        [
            "run",
            "--endpoints",
            f"{trained_simple_project}/endpoints.yml",
            "--model",
            "/models/model.tar.gz",
            "--remote-storage",
            remote_storage,
        ]
    )

    # Run the rasa.cli.run.run function
    cli_run(args)

    # Assert that the arguments are correctly passed to the `rasa run` command
    assert args.remote_storage == expected_remote_storage
    assert args.model == "/models/model.tar.gz"
    assert args.endpoints == f"{trained_simple_project}/endpoints.yml"

    mock_rasa_run.assert_called_once_with(**vars(args))


def test_cli_run_with_skip_yaml_validation(
    run_parser: argparse.ArgumentParser,
    mock_rasa_run: MagicMock,
    trained_simple_project: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests whether the `rasa run` command is invoked to skip Domain YAML validation."""  # noqa: E501
    # Parse the arguments with which Rasa inspect will be run
    args = run_parser.parse_args(
        [
            "run",
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

    # Run the rasa.cli.run.run function
    cli_run(args)

    # Assert that the arguments are correctly passed to the `rasa run` command
    assert args.model == f"{trained_simple_project}/models"
    assert args.endpoints == f"{trained_simple_project}/endpoints.yml"
    assert args.skip_yaml_validation == ["domain"]
    assert not Domain.validate_yaml

    mock_rasa_run.assert_called_once_with(**vars(args))
