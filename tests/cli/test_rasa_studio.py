import argparse
import sys
from typing import TYPE_CHECKING, Callable, Generator
from unittest.mock import Mock

import pytest
from keycloak import KeycloakOpenID
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pytest import CaptureFixture, MonkeyPatch, RunResult

from rasa.cli.studio.studio import _configure_studio_config, _studio_login
from rasa.studio.constants import (
    RASA_STUDIO_AUTH_SERVER_URL_ENV,
    RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV,
    RASA_STUDIO_CLI_DISABLE_VERIFY_KEY_ENV,
    RASA_STUDIO_CLI_REALM_NAME_KEY_ENV,
    RASA_STUDIO_CLI_STUDIO_URL_ENV,
)

if TYPE_CHECKING:
    from prompt_toolkit.input.base import PipeInput


@pytest.fixture
def mock_cli() -> Generator["PipeInput", None, None]:
    pipe_input = create_pipe_input()
    with create_app_session(input=pipe_input, output=DummyOutput()):
        yield pipe_input
    pipe_input.close()


def test_studio_config_help(run: Callable[..., RunResult]):
    output = run("studio", "config", "--help")

    help_text = """usage: rasa studio config [-h] [-v] [-vv] [--quiet]
                 [--logging-config-file LOGGING_CONFIG_FILE]
                 [--disable-verify] [--advanced]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=" '\n' key that accepts the input is not working on Windows.",
)
def test_advanced_asks_for_additional_parameters(mock_cli: "PipeInput") -> None:
    # if this test is hanging, it's likely that the prompt is waiting for input
    # that was not provided - you should provide more input here then
    inputs = ["url", "keycloak", "2", "2", "\n"]
    # use the \n to simulate the user pressing enter
    # see source docs:
    # https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/unit_testing.html#posixpipeinput-and-dummyoutput  # noqa: E501
    inputs = "\n".join(inputs)
    mock_cli.send_text(inputs)

    # if the advanced flag is set, the function should ask for additional parameters
    args: argparse.Namespace = argparse.Namespace(advanced=True, disable_verify=False)

    studio_config = _configure_studio_config(args)
    # the default values are not removed from the input so our input ("2") is
    # just appended to the default values
    assert studio_config.realm_name == "rasa-studio2"
    assert studio_config.client_id == "admin-cli2"
    assert studio_config.authentication_server_url == "https://url/auth/keycloak"
    assert studio_config.studio_url == "https://url/api/graphql/"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=" '\n' key that accepts the input is not working on Windows.",
)
def test_non_advanced_only_asks_for_url(mock_cli: "PipeInput") -> None:
    inputs = ["url", "\n"]
    # use \n to simulate the user pressing enter
    inputs = "\n".join(inputs)
    mock_cli.send_text(inputs)

    # if the advanced flag is not set, the function should only ask for the studio url
    args: argparse.Namespace = argparse.Namespace(advanced=False, disable_verify=False)

    studio_config = _configure_studio_config(args)
    # the default values are not removed from the input so our input ("url") is
    # just appended to the default values
    assert studio_config.realm_name == "rasa-studio"
    assert studio_config.client_id == "admin-cli"
    assert studio_config.authentication_server_url == "https://url/auth/"
    assert studio_config.studio_url == "https://url/api/graphql/"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=" '\n' key that accepts the input is not working on Windows.",
)
def test_non_advanced_only_asks_for_url_disable_verify(
    mock_cli: "PipeInput", capsys: CaptureFixture, monkeypatch: MonkeyPatch
):
    mock_keycloak = Mock(wraps=KeycloakOpenID)
    monkeypatch.setattr("rasa.studio.auth.KeycloakOpenID", mock_keycloak)

    mock_cli.send_text("url\n\n")
    # if the advanced flag is not set, the function should only ask for the studio url
    args: argparse.Namespace = argparse.Namespace(advanced=False, disable_verify=True)

    studio_config = _configure_studio_config(args)
    assert studio_config.disable_verify is True

    captured = capsys.readouterr()
    assert (
        "Disabling SSL verification for the Rasa Studio authentication server."
        in captured.out
    )

    mock_keycloak.assert_called_once_with(
        server_url=studio_config.authentication_server_url,
        client_id=studio_config.client_id,
        realm_name=studio_config.realm_name,
        verify=not studio_config.disable_verify,
    )


def test_studio_download_does_not_throw_endpoints_file_not_found_error(
    run: Callable[..., RunResult],
):
    """Tests that rasa studio commands do not throw endpoints FileNotFound error."""
    error_message = (
        "Failed to read endpoint configuration file - the file was not found."
    )
    output = run("studio", "download", "assistant_name")
    printed_output = {line.strip() for line in output.outlines}

    assert all([error_message not in line for line in printed_output])


@pytest.mark.parametrize("disable_verify", ["true", "false"])
def test_studio_login_reuses_disable_verify_from_studio_config(
    monkeypatch: MonkeyPatch, disable_verify: str
):
    """Assert that the disable_verify flag is reused from the studio config."""
    monkeypatch.setenv(RASA_STUDIO_CLI_STUDIO_URL_ENV, "url")
    monkeypatch.setenv(RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV, "keycloak")
    monkeypatch.setenv(RASA_STUDIO_CLI_REALM_NAME_KEY_ENV, "2")
    monkeypatch.setenv(RASA_STUDIO_CLI_DISABLE_VERIFY_KEY_ENV, disable_verify)
    monkeypatch.setenv(RASA_STUDIO_AUTH_SERVER_URL_ENV, "url/auth/keycloak")

    mock_keycloak = Mock(wraps=KeycloakOpenID)
    monkeypatch.setattr("rasa.studio.auth.KeycloakOpenID", mock_keycloak)

    args: argparse.Namespace = argparse.Namespace(username="user", password="pass")
    _studio_login(args)

    mock_keycloak.assert_called_once_with(
        server_url="url/auth/keycloak",
        client_id="keycloak",
        realm_name="2",
        verify=not bool(disable_verify),
    )
