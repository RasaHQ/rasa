import pytest
import requests

import rasa.utils.io as io_utils
from rasa.cli import x
from rasa.cli.x import _pull_runtime_config_from_server
from rasa.utils.endpoints import EndpointConfig
import responses


def test_x_help(run):
    output = run("x", "--help")

    help_text = """usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--data DATA] [--no-prompt]
              [--production] [--rasa-x-port RASA_X_PORT]
              [--config-endpoint CONFIG_ENDPOINT] [--log-file LOG_FILE]
              [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
              [--cors [CORS [CORS ...]]] [--enable-api]
              [--remote-storage REMOTE_STORAGE] [--credentials CREDENTIALS]
              [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
              [--jwt-method JWT_METHOD]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_prepare_credentials_for_rasa_x_if_rasa_channel_not_given(tmpdir_factory):
    directory = tmpdir_factory.mktemp("directory")
    credentials_path = str(directory / "credentials.yml")

    io_utils.write_yaml_file({}, credentials_path)

    tmp_credentials = x._prepare_credentials_for_rasa_x(
        credentials_path, "http://localhost:5002"
    )

    actual = io_utils.read_config_file(tmp_credentials)

    assert actual["rasa"]["url"] == "http://localhost:5002"


def test_prepare_credentials_if_already_valid(tmpdir_factory):
    directory = tmpdir_factory.mktemp("directory")
    credentials_path = str(directory / "credentials.yml")

    credentials = {
        "rasa": {"url": "my-custom-url"},
        "another-channel": {"url": "some-url"},
    }
    io_utils.write_yaml_file(credentials, credentials_path)

    x._prepare_credentials_for_rasa_x(credentials_path)

    actual = io_utils.read_config_file(credentials_path)

    assert actual == credentials


def test_if_endpoint_config_is_valid_in_local_mode():
    config = EndpointConfig(type="sql", dialect="sqlite", db=x.DEFAULT_TRACKER_DB)

    assert x._is_correct_tracker_store(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "mongo", "url": "mongodb://localhost:27017"},
        {"type": "sql", "dialect": "postgresql"},
        {"type": "sql", "dialect": "sqlite", "db": "some.db"},
    ],
)
def test_if_endpoint_config_is_invalid_in_local_mode(kwargs):
    config = EndpointConfig(**kwargs)

    assert not x._is_correct_tracker_store(config)


@responses.activate
def test_pull_endpoints_and_credentials_from_server():
    config_url = "http://rasa-x.com/api/config?token=token"
    credentials = {"rasa": "http://rasa-x.com:5002/api"}
    endpoints = {
        "event_broker": {
            "url": "http://event-broker.com",
            "username": "some_username",
            "password": "PASSWORRD",
            "queue": "broker_queue",
        }
    }

    responses.add(
        responses.GET,
        config_url,
        json={"credentials": credentials, "endpoints": endpoints},
    )

    endpoints_path, credentials_path = _pull_runtime_config_from_server(
        config_url, 1, 0
    )

    assert endpoints_path
    assert credentials_path

    assert io_utils.read_yaml_file(endpoints_path) == endpoints
    assert io_utils.read_yaml_file(credentials_path) == credentials
