import argparse
from typing import Dict, List

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.cli.arguments.run import (
    add_jwt_arguments,
    add_server_settings_arguments,
    set_run_arguments,
)
from rasa.env import (
    AUTH_TOKEN_ENV,
    DEFAULT_JWT_METHOD,
    JWT_METHOD_ENV,
    JWT_PRIVATE_KEY_ENV,
    JWT_SECRET_ENV,
)

# file deepcode ignore HardcodedNonCryptoSecret/test: Test credentials


@pytest.mark.parametrize(
    "env_variables, input_args, expected",
    [
        (
            # all env variables are set
            {
                JWT_SECRET_ENV: "secret",
                JWT_METHOD_ENV: "HS256",
                JWT_PRIVATE_KEY_ENV: "private_key",
            },
            [],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method="HS256",
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_SECRET_ENV and --jwt-secret is set
            {
                JWT_METHOD_ENV: "HS256",
                JWT_PRIVATE_KEY_ENV: "private_key",
            },
            ["--jwt-secret", "secret"],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method="HS256",
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_METHOD_ENV and --jwt-method is set
            {
                JWT_SECRET_ENV: "secret",
                JWT_PRIVATE_KEY_ENV: "private_key",
            },
            ["--jwt-method", "HS256"],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method="HS256",
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_PRIVATE_KEY_ENV and --jwt-private-key is set
            {
                JWT_SECRET_ENV: "secret",
                JWT_METHOD_ENV: "HS256",
            },
            ["--jwt-private-key", "private_key"],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method="HS256",
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_SECRET_ENV and no --jwt-secret
            {
                JWT_METHOD_ENV: "HS256",
                JWT_PRIVATE_KEY_ENV: "private_key",
            },
            [],
            argparse.Namespace(
                jwt_secret=None,
                jwt_method="HS256",
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_METHOD_ENV and no --jwt-method
            {
                JWT_SECRET_ENV: "secret",
                JWT_PRIVATE_KEY_ENV: "private_key",
            },
            [],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method=DEFAULT_JWT_METHOD,
                jwt_private_key="private_key",
            ),
        ),
        (
            # no JWT_PRIVATE_KEY_ENV and no --jwt-private-key
            {
                JWT_SECRET_ENV: "secret",
                JWT_METHOD_ENV: "HS256",
            },
            [],
            argparse.Namespace(
                jwt_secret="secret",
                jwt_method="HS256",
                jwt_private_key=None,
            ),
        ),
        (
            # no env variables and no arguments
            {},
            [],
            argparse.Namespace(
                jwt_secret=None,
                jwt_method="HS256",
                jwt_private_key=None,
            ),
        ),
    ],
)
def test_jwt_argument_parsing(
    env_variables: Dict[str, str],
    input_args: List[str],
    expected: argparse.Namespace,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests parsing of the JWT arguments."""
    parser = argparse.ArgumentParser()

    for env_name, env_value in env_variables.items():
        monkeypatch.setenv(env_name, env_value)

    add_jwt_arguments(parser)
    args = parser.parse_args(input_args)

    assert args.jwt_secret == expected.jwt_secret
    assert args.jwt_method == expected.jwt_method
    assert args.jwt_private_key == expected.jwt_private_key


@pytest.mark.parametrize(
    "env_variables, input_args, expected",
    [
        (
            {
                AUTH_TOKEN_ENV: "secret",
            },
            [],
            argparse.Namespace(
                auth_token="secret",
            ),
        ),
        (
            {},
            ["--auth-token", "secret"],
            argparse.Namespace(
                auth_token="secret",
            ),
        ),
        (
            {},
            [],
            argparse.Namespace(
                auth_token=None,
            ),
        ),
    ],
)
def test_add_server_settings_arguments(
    env_variables: Dict[str, str],
    input_args: List[str],
    expected: argparse.Namespace,
    monkeypatch: MonkeyPatch,
) -> None:
    """Tests parsing of the server settings arguments."""
    parser = argparse.ArgumentParser()

    for env_name, env_value in env_variables.items():
        monkeypatch.setenv(env_name, env_value)

    add_server_settings_arguments(parser)

    args = parser.parse_args(input_args)

    assert args.auth_token == expected.auth_token


@pytest.mark.parametrize(
    "input_args, expected_skip_yaml_validation",
    [
        (
            [],
            [],
        ),
        (
            ["--skip-yaml-validation", "domain"],
            ["domain"],
        ),
    ],
)
def test_run_cli_skip_yaml_validation_flag(
    input_args: List[str], expected_skip_yaml_validation: List[str]
) -> None:
    """Tests that --skip-yaml-validation is attached to the run CLI."""
    parser = argparse.ArgumentParser()

    set_run_arguments(parser)

    args = parser.parse_args(input_args)

    assert args.skip_yaml_validation == expected_skip_yaml_validation


def test_default_run_arguments(
    run_parser: argparse.ArgumentParser,
) -> None:
    """Tests default settings for `rasa run` CLI command."""
    args = run_parser.parse_args(["run"])

    assert args.model == "models"
    assert args.log_file is None
    assert not args.use_syslog
    assert args.syslog_address == "localhost"
    assert args.syslog_port == 514
    assert args.syslog_protocol == "UDP"
    assert args.endpoints == "endpoints.yml"
    assert args.interface == "0.0.0.0"
    assert args.port == 5005
    assert args.auth_token is None
    assert args.cors is None
    assert not args.enable_api
    assert args.response_timeout == 3600
    assert args.request_timeout == 300
    assert args.remote_storage is None
    assert args.ssl_certificate is None
    assert args.ssl_keyfile is None
    assert args.ssl_ca_file is None
    assert args.ssl_password is None
    assert args.credentials is None
    assert args.connector is None
    assert args.jwt_secret is None
    assert args.jwt_method == "HS256"
    assert args.jwt_private_key is None
    assert args.skip_yaml_validation == []
    assert args.sub_agents == "sub_agents"
    assert args.legacy is False


def test_run_inspect_legacy_flag(
    run_parser: argparse.ArgumentParser,
) -> None:
    """Tests that --legacy flag is parsed correctly for rasa run --inspect --legacy."""
    args = run_parser.parse_args(["run", "--inspect", "--legacy"])

    assert args.inspect is True
    assert args.legacy is True
