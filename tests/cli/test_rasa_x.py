from pathlib import Path
import logging

import pytest
from typing import Callable, Dict
from _pytest.pytester import RunResult
from _pytest.logging import LogCaptureFixture


from aioresponses import aioresponses

import rasa.utils.io as io_utils
from rasa.cli import x
from rasa.utils.endpoints import EndpointConfig
from rasa.core.utils import AvailableEndpoints
from tests.conftest import assert_log_emitted


def test_x_help(run: Callable[..., RunResult]):
    output = run("x", "--help")

    help_text = """usage: rasa x [-h] [-v] [-vv] [--quiet] [-m MODEL] [--data DATA] [--no-prompt]
              [--production] [--rasa-x-port RASA_X_PORT]
              [--config-endpoint CONFIG_ENDPOINT] [--log-file LOG_FILE]
              [--endpoints ENDPOINTS] [-p PORT] [-t AUTH_TOKEN]
              [--cors [CORS [CORS ...]]] [--enable-api]
              [--remote-storage REMOTE_STORAGE]
              [--ssl-certificate SSL_CERTIFICATE] [--ssl-keyfile SSL_KEYFILE]
              [--ssl-ca-file SSL_CA_FILE] [--ssl-password SSL_PASSWORD]
              [--credentials CREDENTIALS] [--connector CONNECTOR]
              [--jwt-secret JWT_SECRET] [--jwt-method JWT_METHOD]"""

    lines = help_text.split("\n")

    for i, line in enumerate(lines):
        assert output.outlines[i] == line


def test_prepare_credentials_for_rasa_x_if_rasa_channel_not_given(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    io_utils.write_yaml_file({}, credentials_path)

    tmp_credentials = x._prepare_credentials_for_rasa_x(
        credentials_path, "http://localhost:5002"
    )

    actual = io_utils.read_config_file(tmp_credentials)

    assert actual["rasa"]["url"] == "http://localhost:5002"


def test_prepare_credentials_if_already_valid(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    credentials = {
        "rasa": {"url": "my-custom-url"},
        "another-channel": {"url": "some-url"},
    }
    io_utils.write_yaml_file(credentials, credentials_path)

    x._prepare_credentials_for_rasa_x(credentials_path)

    actual = io_utils.read_config_file(credentials_path)

    assert actual == credentials


def test_if_endpoint_config_is_valid_in_local_mode():
    config = EndpointConfig(type="sql", dialect="sqlite", db=x.DEFAULT_EVENTS_DB)

    assert x._is_correct_event_broker(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"type": "mongo", "url": "mongodb://localhost:27017"},
        {"type": "sql", "dialect": "postgresql"},
        {"type": "sql", "dialect": "sqlite", "db": "some.db"},
    ],
)
def test_if_endpoint_config_is_invalid_in_local_mode(kwargs: Dict):
    config = EndpointConfig(**kwargs)
    assert not x._is_correct_event_broker(config)


def test_overwrite_for_local_x(caplog: LogCaptureFixture):
    test_wait_time = 5
    default_wait_time = 2
    endpoint_config_missing_wait = EndpointConfig(
        url="http://testserver:5002/models/default@latest"
    )
    endpoint_config_custom = EndpointConfig(
        url="http://testserver:5002/models/default@latest",
        wait_time_between_pulls=test_wait_time,
    )
    endpoints_custom = AvailableEndpoints(model=endpoint_config_custom)
    endpoints_missing_wait = AvailableEndpoints(model=endpoint_config_missing_wait)

    # Check that we get INFO message about overwriting the endpoints configuration
    log_message = "Ignoring url 'http://testserver:5002/models/default@latest' from 'endpoints.yml' and using 'http://localhost/projects/default/models/tag/production' instead"
    with assert_log_emitted(caplog, "rasa.cli.x", logging.INFO, log_message):
        x._overwrite_endpoints_for_local_x(endpoints_custom, "test", "http://localhost")

    # Checking for url to be changed in config and wait time value to be honored
    assert (
        endpoints_custom.model.url
        == "http://localhost/projects/default/models/tag/production"
    )
    assert endpoints_custom.model.kwargs["wait_time_between_pulls"] == test_wait_time

    # Check for wait time to be set to 3 since it isn't specified
    x._overwrite_endpoints_for_local_x(
        endpoints_missing_wait, "test", "http://localhost"
    )
    assert (
        endpoints_missing_wait.model.kwargs["wait_time_between_pulls"]
        == default_wait_time
    )


async def test_pull_runtime_config_from_server():
    config_url = "http://example.com/api/config?token=token"
    credentials = "rasa: http://example.com:5002/api"
    endpoint_config = """
    event_broker:
        url: http://example.com/event_broker
        username: some_username
        password: PASSWORD
        queue: broker_queue
    """
    with aioresponses() as mocked:
        mocked.get(
            config_url,
            payload={"credentials": credentials, "endpoints": endpoint_config},
        )

        endpoints_path, credentials_path = await x._pull_runtime_config_from_server(
            config_url, 1, 0
        )

        with open(endpoints_path) as f:
            assert f.read() == endpoint_config
        with open(credentials_path) as f:
            assert f.read() == credentials
