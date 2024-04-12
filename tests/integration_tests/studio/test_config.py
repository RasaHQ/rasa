from pathlib import Path
from typing import Dict, Text

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    RASA_STUDIO_AUTH_SERVER_URL_ENV,
    RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV,
    RASA_STUDIO_CLI_REALM_NAME_KEY_ENV,
    RASA_STUDIO_CLI_STUDIO_URL_ENV,
)


@pytest.fixture
def mock_config_path(tmp_path: Path) -> Path:
    return tmp_path / "global.yml"


@pytest.fixture
def mock_global_config_path(mock_config_path: Path, monkeypatch: MonkeyPatch) -> None:
    print(mock_config_path)
    monkeypatch.setattr("rasa.constants.GLOBAL_USER_CONFIG_PATH", str(mock_config_path))


@pytest.mark.usefixtures("mock_global_config_path")
def test_studio_config_read_empty_file() -> None:
    config = StudioConfig.read_config()
    assert config.is_valid() is False


@pytest.mark.usefixtures("mock_global_config_path")
def test_studio_config_read_and_write_file() -> None:
    config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )
    config.write_config()
    read_config = StudioConfig.read_config()
    assert config == read_config


@pytest.mark.usefixtures("mock_global_config_path")
@pytest.mark.parametrize(
    "env, file, expected",
    [
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://newlocalhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://localhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
            StudioConfig(
                authentication_server_url="http://newlocalhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://localhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "new_client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="new_client_id",
                realm_name="realm_name",
            ),
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://localhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "new_realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="new_realm_name",
            ),
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://newlocalhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "new_realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://newlocalhost:8080/graphql",
                client_id="client_id",
                realm_name="new_realm_name",
            ),
        ),
    ],
)
def test_studio_config_read_from_file_and_env(
    env: Dict[Text, Text],
    file: StudioConfig,
    expected: StudioConfig,
    monkeypatch: MonkeyPatch,
) -> None:
    file.write_config()

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = StudioConfig.read_config()
    assert config == expected
