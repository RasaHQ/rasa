# file deepcode ignore HardcodedNonCryptoSecret/test: Secrets are all just examples for tests. # noqa: E501

from typing import Dict, Text
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.studio.auth import (
    ACCESS_TOKEN_EXPIRATION_TIME_KEY,
    ACCESS_TOKEN_KEY,
    DEFAULT_TOKEN_FILE_PATH,
    REFRESH_TOKEN_EXPIRATION_TIME_KEY,
    REFRESH_TOKEN_KEY,
    KeycloakToken,
    KeycloakTokenReader,
    KeycloakTokenWriter,
    StudioAuth,
)
from rasa.studio.config import StudioConfig
from rasa.studio.constants import (
    KEYCLOAK_ACCESS_TOKEN_KEY,
    KEYCLOAK_EXPIRES_IN_KEY,
    KEYCLOAK_REFRESH_EXPIRES_IN_KEY,
    KEYCLOAK_REFRESH_TOKEN,
    KEYCLOAK_TOKEN_TYPE,
)


@pytest.fixture
def mock_token_path(monkeypatch: MonkeyPatch) -> MagicMock:
    token_path = MagicMock()
    token_path.is_file().return_value = True
    return token_path


@pytest.fixture
def mock_read_yaml_file(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.auth.read_yaml_file", mock)
    return mock


@pytest.fixture
def mock_write_yaml(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.auth.write_yaml", mock)
    return mock


@pytest.fixture
def mock_write_token_to_file(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(
        "rasa.studio.auth.KeycloakTokenWriter.write_token_to_file", mock
    )
    return mock


@pytest.fixture
def mock_keycloak_open_id(monkeypatch: MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr("rasa.studio.auth.KeycloakOpenID", mock)
    return mock


def test_studio_auth(mock_keycloak_instance: MagicMock) -> None:
    config = StudioConfig(
        authentication_server_url="http://localhost:8080",
        studio_url="http://localhost:8080/graphql",
        client_id="client_id",
        realm_name="realm_name",
    )
    auth = StudioAuth(config)

    assert auth.config == config
    assert auth.keycloak_openid == mock_keycloak_instance


def test_studio_auth_login(
    mock_keycloak_instance: MagicMock,
    mock_write_token_to_file: MagicMock,
    studio_auth: StudioAuth,
) -> None:
    token_dict = {
        KEYCLOAK_ACCESS_TOKEN_KEY: "access_token",
        KEYCLOAK_EXPIRES_IN_KEY: 1800,
        KEYCLOAK_REFRESH_EXPIRES_IN_KEY: 2400,
        KEYCLOAK_REFRESH_TOKEN: "refresh_token",
        KEYCLOAK_TOKEN_TYPE: "Bearer",
    }

    token = KeycloakToken(
        access_token=token_dict[KEYCLOAK_ACCESS_TOKEN_KEY],  # type: ignore
        expires_in=token_dict[KEYCLOAK_EXPIRES_IN_KEY],  # type: ignore
        refresh_expires_in=token_dict[KEYCLOAK_REFRESH_EXPIRES_IN_KEY],  # type: ignore
        refresh_token=token_dict[KEYCLOAK_REFRESH_TOKEN],  # type: ignore
        token_type=token_dict[KEYCLOAK_TOKEN_TYPE],  # type: ignore
    )

    mock_keycloak_instance.token.return_value = token_dict
    studio_auth.login("username", "password", 1234)

    mock_keycloak_instance.token.assert_called_once_with(
        username="username", password="password", totp=1234
    )

    mock_write_token_to_file.assert_called_once_with(
        token, token_file_location=DEFAULT_TOKEN_FILE_PATH
    )


def test_keycloak_token_to_dict() -> None:
    token = KeycloakToken(
        access_token="access_token",
        expires_in=1800,
        refresh_expires_in=2400,
        refresh_token="refresh_token",
        token_type="token_type",
    )

    result = token.to_dict()

    assert result == {
        ACCESS_TOKEN_KEY: token.access_token,
        ACCESS_TOKEN_EXPIRATION_TIME_KEY: token.expires_in,
        REFRESH_TOKEN_KEY: token.refresh_token,
        REFRESH_TOKEN_EXPIRATION_TIME_KEY: token.refresh_expires_in,
        KEYCLOAK_TOKEN_TYPE: token.token_type,
    }


@pytest.mark.parametrize(
    "expected",
    [
        KeycloakToken(
            access_token="access_token",
            expires_in=21600,
            refresh_expires_in=1800,
            refresh_token="refresh_token",
            token_type="Bearer",
        )
    ],
)
def test_keycloak_token_reader(
    expected: KeycloakToken, mock_read_yaml_file: MagicMock, mock_token_path: MagicMock
) -> None:
    mock_read_yaml_file.return_value = expected.to_dict()

    reader = KeycloakTokenReader(token_path=mock_token_path)

    assert reader.get_token() == expected
    mock_read_yaml_file.assert_called_once_with(mock_token_path)


@pytest.mark.parametrize(
    "input_token_dict",
    [
        {},
        {
            ACCESS_TOKEN_KEY: None,
        },
        {
            ACCESS_TOKEN_KEY: "",
        },
    ],
)
def test_keycloak_token_reader_no_access_token(
    input_token_dict: Dict[Text, Text],
    mock_read_yaml_file: MagicMock,
    mock_token_path: MagicMock,
) -> None:
    mock_read_yaml_file.return_value = input_token_dict

    with pytest.raises(ValueError):
        KeycloakTokenReader(token_path=mock_token_path)

    mock_read_yaml_file.assert_called_once_with(mock_token_path)


@pytest.mark.parametrize(
    "input_token_dict",
    [
        {ACCESS_TOKEN_KEY: "some token"},
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: None,
        },
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "",
        },
    ],
)
def test_keycloak_token_reader_no_access_token_expiration_time(
    input_token_dict: Dict[Text, Text],
    mock_read_yaml_file: MagicMock,
    mock_token_path: MagicMock,
) -> None:
    mock_read_yaml_file.return_value = input_token_dict

    with pytest.raises(ValueError):
        KeycloakTokenReader(token_path=mock_token_path)

    mock_read_yaml_file.assert_called_once_with(mock_token_path)


@pytest.mark.parametrize(
    "input_token_dict",
    [
        {ACCESS_TOKEN_KEY: "some token", ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time"},
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time",
            REFRESH_TOKEN_KEY: None,
        },
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time",
            REFRESH_TOKEN_KEY: "",
        },
    ],
)
def test_keycloak_token_reader_no_refresh_token(
    input_token_dict: Dict[Text, Text],
    mock_read_yaml_file: MagicMock,
    mock_token_path: MagicMock,
) -> None:
    mock_read_yaml_file.return_value = input_token_dict

    with pytest.raises(ValueError):
        KeycloakTokenReader(token_path=mock_token_path)

    mock_read_yaml_file.assert_called_once_with(mock_token_path)


@pytest.mark.parametrize(
    "input_token_dict",
    [
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time",
            REFRESH_TOKEN_KEY: "refresh token",
        },
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time",
            REFRESH_TOKEN_KEY: "refresh token",
            REFRESH_TOKEN_EXPIRATION_TIME_KEY: None,
        },
        {
            ACCESS_TOKEN_KEY: "some token",
            ACCESS_TOKEN_EXPIRATION_TIME_KEY: "some time",
            REFRESH_TOKEN_KEY: "refresh token",
            REFRESH_TOKEN_EXPIRATION_TIME_KEY: "",
        },
    ],
)
def test_keycloak_token_reader_no_refresh_token_expiration_time(
    input_token_dict: Dict[Text, Text],
    mock_read_yaml_file: MagicMock,
    mock_token_path: MagicMock,
) -> None:
    mock_read_yaml_file.return_value = input_token_dict

    with pytest.raises(ValueError):
        KeycloakTokenReader(token_path=mock_token_path)

    mock_read_yaml_file.assert_called_once_with(mock_token_path)


def test_write_token_to_file(mock_write_yaml: MagicMock) -> None:
    token = KeycloakToken(
        access_token="access_token",
        expires_in=1800,
        refresh_expires_in=2400,
        refresh_token="refresh_token",
        token_type="Bearer",
    )
    path = DEFAULT_TOKEN_FILE_PATH

    KeycloakTokenWriter.write_token_to_file(
        token=token,
        token_file_location=path,
    )

    mock_write_yaml.assert_called_once_with(token.to_dict(), path)
