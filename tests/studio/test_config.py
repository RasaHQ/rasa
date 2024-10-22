from typing import Dict, Text
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    RASA_STUDIO_AUTH_SERVER_URL_ENV,
    RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV,
    RASA_STUDIO_CLI_REALM_NAME_KEY_ENV,
    RASA_STUDIO_CLI_STUDIO_URL_ENV,
    STUDIO_CONFIG_KEY,
)


@pytest.fixture
def mock_write_global_config_value(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.config.write_global_config_value", mock)
    return mock


@pytest.fixture
def mock_read_global_config_value(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.config.read_global_config_value", mock)
    return mock


@pytest.fixture
def mock_os_makedirs(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa_plus.studio.config.os.makedirs", mock)
    return mock


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
        ),
    ],
)
def test_studio_config_to_dict(
    config: StudioConfig, expected: Dict[Text, Text]
) -> None:
    assert config.to_dict() == expected


@pytest.mark.parametrize(
    "studio_dict_input, expected",
    [
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
    ],
)
def test_studio_config_from_dict(
    studio_dict_input: Dict[Text, Text], expected: StudioConfig
) -> None:
    assert StudioConfig.from_dict(studio_dict_input) == expected


def test_studio_config_write_config(mock_write_global_config_value: MagicMock) -> None:
    studio_config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )

    studio_config.write_config()

    mock_write_global_config_value.assert_called_once_with(
        STUDIO_CONFIG_KEY, studio_config.to_dict()
    )


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
    ],
)
def test_studio_config_read_config_from_file(
    input_dict: Dict[Text, Text],
    expected: StudioConfig,
    mock_read_global_config_value: MagicMock,
) -> None:
    mock_read_global_config_value.return_value = input_dict

    result = StudioConfig.read_config()

    mock_read_global_config_value.assert_called_once_with(
        STUDIO_CONFIG_KEY, unavailable_ok=True
    )

    assert result == expected


@pytest.mark.parametrize(
    "file_content",
    [
        (
            {
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "realm_name": "realm_name",
            },
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "client_id": "client_id",
            },
        ),
        (
            {
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
            },
        ),
    ],
)
def test_studio_config_read_config_from_file_no_file(
    file_content: Dict[Text, Text],
    mock_read_global_config_value: MagicMock,
) -> None:
    mock_read_global_config_value.exists.return_value = file_content

    with pytest.raises(ValueError):
        StudioConfig.read_config()

    mock_read_global_config_value.assert_called_once_with(
        STUDIO_CONFIG_KEY, unavailable_ok=True
    )


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
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
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: None,
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://localhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "realm_name",
            },
            StudioConfig(
                authentication_server_url=None,
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: None,
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: None,
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "realm_name",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url=None,
                client_id=None,
                realm_name="realm_name",
            ),
        ),
        (
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://localhost:8080",
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://localhost:8080/graphql",
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id",
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: None,
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name=None,
            ),
        ),
    ],
)
def test_studio_config_read_config_from_env(
    input_dict: Dict[Text, Text],
    expected: StudioConfig,
    mock_read_global_config_value: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_global_config_value.return_value = None

    for key, value in input_dict.items():
        if value is not None:
            monkeypatch.setenv(key, value)

    result = StudioConfig.read_config()

    assert result == expected

    mock_read_global_config_value.assert_called_once_with(
        STUDIO_CONFIG_KEY, unavailable_ok=True
    )


@pytest.mark.parametrize(
    "file_content, env, expected",
    [
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            {},
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            {
                RASA_STUDIO_AUTH_SERVER_URL_ENV: "http://newlocalhost:8080",
            },
            StudioConfig(
                authentication_server_url="http://newlocalhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            {
                RASA_STUDIO_CLI_CLIENT_ID_KEY_ENV: "client_id_env",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id_env",
                realm_name="realm_name",
            ),
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            {
                RASA_STUDIO_CLI_REALM_NAME_KEY_ENV: "realm_name_env",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://localhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name_env",
            ),
        ),
        (
            {
                "authentication_server_url": "http://localhost:8080",
                "studio_url": "http://localhost:8080/graphql",
                "client_id": "client_id",
                "realm_name": "realm_name",
            },
            {
                RASA_STUDIO_CLI_STUDIO_URL_ENV: "http://newlocalhost:8080/graphql",
            },
            StudioConfig(
                authentication_server_url="http://localhost:8080",
                studio_url="http://newlocalhost:8080/graphql",
                client_id="client_id",
                realm_name="realm_name",
            ),
        ),
    ],
)
def test_studio_config_read_config(
    file_content: Dict[Text, Text],
    env: Dict[Text, Text],
    expected: StudioConfig,
    mock_read_global_config_value: MagicMock,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_global_config_value.return_value = file_content

    for key, value in env.items():
        monkeypatch.setenv(key, value)

    result = StudioConfig.read_config()

    mock_read_global_config_value.assert_called_once_with(
        STUDIO_CONFIG_KEY, unavailable_ok=True
    )

    assert result == expected
