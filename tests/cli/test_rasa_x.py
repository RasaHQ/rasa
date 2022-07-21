from pathlib import Path
import sys
import argparse

import pytest
from typing import Callable

from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import RunResult
from aioresponses import aioresponses

import rasa.shared.utils.io
from rasa.cli import x
from rasa.utils.endpoints import EndpointConfig
from rasa.core.utils import AvailableEndpoints
import rasa.version

from tests.cli.conftest import RASA_EXE


def test_x_help(run: Callable[..., RunResult]):
    output = run("x", "--help")

    if sys.version_info.minor >= 9:
        # This is required because `argparse` behaves differently on
        # Python 3.9 and above. The difference is the changed formatting of help
        # output for CLI arguments with `nargs="*"
        version_dependent = [
            "[-i INTERFACE]",
            "[-p PORT]",
            "[-t AUTH_TOKEN]",
            "[--cors [CORS ...]]",
            "[--enable-api]",
            "[--response-timeout RESPONSE_TIMEOUT]",
        ]
    else:
        version_dependent = [
            "[-i INTERFACE]",
            "[-p PORT]",
            "[-t AUTH_TOKEN]",
            "[--cors [CORS [CORS ...]]]",
            "[--enable-api]",
            "[--response-timeout RESPONSE_TIMEOUT]",
        ]

    help_text = (
        [
            f"{RASA_EXE} x",
            "[-h]",
            "[-v]",
            "[-vv]",
            "[--quiet]",
            "[-m MODEL]",
            "[--no-prompt]",
            "[--production]",
            "[--config-endpoint CONFIG_ENDPOINT]",
            "[--log-file LOG_FILE]",
            "[--use-syslog]",
            "[--syslog-address SYSLOG_ADDRESS]",
            "[--syslog-port SYSLOG_PORT]",
            "[--syslog-protocol SYSLOG_PROTOCOL]",
            "[--endpoints ENDPOINTS]",
        ]
        + version_dependent
        + [
            "--remote-storage REMOTE_STORAGE]",
            "[--ssl-certificate SSL_CERTIFICATE]",
            "[--ssl-keyfile SSL_KEYFILE]",
            "[--ssl-ca-file SSL_CA_FILE]",
            "[--ssl-password SSL_PASSWORD]",
            "[--credentials CREDENTIALS]",
            "[--connector CONNECTOR]",
            "[--jwt-secret JWT_SECRET]",
            "[--jwt-method JWT_METHOD]",
        ]
    )

    # expected help text lines should appear somewhere in the output
    printed_help = " ".join(output.outlines)
    for item in help_text:
        assert item in printed_help


def test_prepare_credentials_for_rasa_x_if_rasa_channel_not_given(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    rasa.shared.utils.io.write_yaml({}, credentials_path)

    tmp_credentials = x._prepare_credentials_for_rasa_x(
        credentials_path, "http://localhost:5002"
    )

    actual = rasa.shared.utils.io.read_config_file(tmp_credentials)

    assert actual["rasa"]["url"] == "http://localhost:5002"


def test_prepare_credentials_if_already_valid(tmpdir: Path):
    credentials_path = str(tmpdir / "credentials.yml")

    credentials = {
        "rasa": {"url": "my-custom-url"},
        "another-channel": {"url": "some-url"},
    }
    rasa.shared.utils.io.write_yaml(credentials, credentials_path)

    x._prepare_credentials_for_rasa_x(credentials_path)

    actual = rasa.shared.utils.io.read_config_file(credentials_path)

    assert actual == credentials


def test_reuse_wait_time_between_pulls():
    test_wait_time = 5
    endpoint_config = EndpointConfig(
        url="http://localhost:5002/models/default@latest",
        wait_time_between_pulls=test_wait_time,
    )
    endpoints = AvailableEndpoints(model=endpoint_config)
    assert endpoints.model.kwargs["wait_time_between_pulls"] == test_wait_time


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

        assert rasa.shared.utils.io.read_file(endpoints_path) == endpoint_config
        assert rasa.shared.utils.io.read_file(credentials_path) == credentials


def test_rasa_x_raises_warning_and_exits_without_production_flag():

    args = argparse.Namespace(loglevel=None, log_file=None, production=None)
    with pytest.raises(SystemExit):
        with pytest.warns(
            UserWarning,
            match="Running Rasa X in local mode is no longer supported as Rasa has "
            "stopped supporting the Community Edition (free version) of ‘Rasa X’. "
            "For more information please see "
            "https://rasa.com/blog/rasa-x-community-edition-changes/",
        ):
            x.rasa_x(args)


def test_rasa_x_does_not_raise_warning_with_production_flag(
    monkeypatch: MonkeyPatch,
):
    def mock_run_in_enterprise_connection_mode(args):
        return None

    monkeypatch.setattr(
        x, "run_in_enterprise_connection_mode", mock_run_in_enterprise_connection_mode
    )

    args = argparse.Namespace(loglevel=None, log_file=None, production=True)

    with pytest.warns(None):
        x.rasa_x(args)
