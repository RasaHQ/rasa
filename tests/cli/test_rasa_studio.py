import argparse
from typing import Callable

from pytest import RunResult
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


from rasa.cli.studio.studio import _configure_studio_config


@pytest.fixture
def mock_cli():
    pipe_input = create_pipe_input()
    with create_app_session(input=pipe_input, output=DummyOutput()):
        yield pipe_input
    pipe_input.close()


def test_studio_config_help(run: Callable[..., RunResult]):
    output = run("studio", "config", "--help")

    help_text = """usage: rasa studio config [-h] [-v] [-vv] [--quiet]
                 [--logging-config-file LOGGING_CONFIG_FILE]
                 [--advanced]"""

    lines = help_text.split("\n")
    # expected help text lines should appear somewhere in the output
    printed_help = {line.strip() for line in output.outlines}
    for line in lines:
        assert line.strip() in printed_help


def test_advanced_asks_for_additional_parameters(mock_cli):
    # if this test is hanging, it's likely that the prompt is waiting for input
    # that was not provided - you should provide more input here then
    mock_cli.send_text("url\nkeycloak\n2\n2\n\n")
    # if the advanced flag is set, the function should ask for additional parameters
    args: argparse.Namespace = argparse.Namespace(advanced=True)

    studio_config = _configure_studio_config(args)
    # the default values are not removed from the input so our input ("2") is
    # just appended to the default values
    assert studio_config.realm_name == "rasa-studio2"
    assert studio_config.client_id == "admin-cli2"
    assert studio_config.authentication_server_url == "https://url/auth/keycloak"
    assert studio_config.studio_url == "https://url/api/graphql/"


def test_non_advanced_only_asks_for_url(mock_cli):
    mock_cli.send_text("url\n\n")
    # if the advanced flag is not set, the function should only ask for the studio url
    args: argparse.Namespace = argparse.Namespace(advanced=False)

    studio_config = _configure_studio_config(args)
    # the default values are not removed from the input so our input ("url") is
    # just appended to the default values
    assert studio_config.realm_name == "rasa-studio"
    assert studio_config.client_id == "admin-cli"
    assert studio_config.authentication_server_url == "https://url/auth/"
    assert studio_config.studio_url == "https://url/api/graphql/"


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
