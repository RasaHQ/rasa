# file deepcode ignore HardcodedNonCryptoSecret/test: Secrets are all just examples for tests. # noqa: E501

from typing import Any, Dict, Optional, Text
from unittest.mock import MagicMock

import pytest
from pytest import MonkeyPatch
from rasa.shared.exceptions import RasaException
from rasa.utils.endpoints import EndpointConfig

from rasa.core.secrets_manager.constants import (
    SECRET_MANAGER_ENV_NAME,
    TRANSIT_MOUNT_POINT_LABEL,
    VAULT_NAMESPACE_ENV_NAME,
    VAULT_RASA_SECRETS_PATH_ENV_NAME,
    VAULT_TOKEN_ENV_NAME,
    VAULT_TRANSIT_MOUNT_POINT_ENV_NAME,
    VAULT_URL_ENV_NAME,
)
from rasa.core.secrets_manager.factory import (
    load_secret_manager,
    read_secret_manager_config,
    read_vault_endpoint_config,
)
from rasa.core.secrets_manager.secret_manager import (
    SecretManagerConfig,
    SecretsManager,
    SecretsManagerProvider,
)
from rasa.core.secrets_manager.vault import VaultSecretManagerConfig


@pytest.fixture
def mock_read_endpoint_config(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_read_endpoint_config = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.factory.read_endpoint_config",
        _mock_read_endpoint_config,
    )
    return _mock_read_endpoint_config


@pytest.mark.parametrize(
    "endpoint_config, env_vars, expected_result",
    [
        (
            None,
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                url="http://localhost:8200",
            ),
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                token="some_token",
            ),
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                secrets_path="rasa-secrets2",
            ),
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                transit_mount_point="some transit key",
            ),
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                type="vault",
            ),
            {
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
                VAULT_NAMESPACE_ENV_NAME: "admin/mynamespace",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                namespace="admin/mynamespace",
            ),
            {
                SECRET_MANAGER_ENV_NAME: "vault",
                VAULT_URL_ENV_NAME: "http://localhost:8200",
                VAULT_TOKEN_ENV_NAME: "some_token",
                VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
                VAULT_TRANSIT_MOUNT_POINT_ENV_NAME: "some transit key",
            },
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
        (
            EndpointConfig(
                type="vault",
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
            {},
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
                namespace="admin/mynamespace",
            ),
        ),
    ],
)
def test_read_secret_manager_config_with_endpoint_config(
    mock_read_endpoint_config: Any,
    endpoint_config: Optional[Dict],
    env_vars: Dict,
    expected_result: VaultSecretManagerConfig,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_endpoint_config.return_value = endpoint_config

    for key in env_vars.keys():
        monkeypatch.setenv(key, env_vars[key])

    result = read_secret_manager_config(endpoints_file="some file")

    assert result is not None
    assert isinstance(result, VaultSecretManagerConfig)
    assert result.secret_manager_type == expected_result.secret_manager_type
    assert result.url == expected_result.url
    assert result.token == expected_result.token
    assert result.secrets_path == expected_result.secrets_path
    assert result.transit_mount_point == expected_result.transit_mount_point
    assert result.namespace == expected_result.namespace


@pytest.mark.parametrize(
    "env_vars",
    [
        {  # no url
            SECRET_MANAGER_ENV_NAME: "vault",
            VAULT_TOKEN_ENV_NAME: "some_token",
            VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
            TRANSIT_MOUNT_POINT_LABEL: "some transit key",
        },
        {  # no token
            SECRET_MANAGER_ENV_NAME: "vault",
            VAULT_URL_ENV_NAME: "http://localhost:8200",
            VAULT_RASA_SECRETS_PATH_ENV_NAME: "rasa-secrets2",
            TRANSIT_MOUNT_POINT_LABEL: "some transit key",
        },
        {  # no secrets_path
            SECRET_MANAGER_ENV_NAME: "vault",
            VAULT_URL_ENV_NAME: "http://localhost:8200",
            VAULT_TOKEN_ENV_NAME: "some_token",
            TRANSIT_MOUNT_POINT_LABEL: "some transit key",
        },
    ],
)
def test_read_secret_manager_config_with_endpoint_config_raise_exception(
    mock_read_endpoint_config: MagicMock,
    env_vars: Dict[Text, Text],
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_endpoint_config.return_value = None

    for key in env_vars.keys():
        monkeypatch.setenv(key, env_vars[key])

    with pytest.raises(RasaException):
        read_secret_manager_config(endpoints_file="some_file")


def test_read_secret_manager_config_with_endpoint_config_none() -> None:
    assert read_secret_manager_config(endpoints_file=None) is None


@pytest.fixture
def mock_create(monkeypatch: MonkeyPatch) -> MagicMock:
    _mock_create = MagicMock()
    monkeypatch.setattr("rasa.core.secrets_manager.factory.create", _mock_create)
    return _mock_create


@pytest.fixture
def mock_read_secret_manager_config(monkeypatch: MonkeyPatch) -> SecretsManager:
    _mock_read_secret_manager_config = MagicMock()
    monkeypatch.setattr(
        "rasa.core.secrets_manager.factory.read_secret_manager_config",
        _mock_read_secret_manager_config,
    )
    return _mock_read_secret_manager_config


@pytest.mark.parametrize(
    "config, secrets_manager ",
    [
        (
            VaultSecretManagerConfig(
                url="http://localhost:8200",
                token="some_token",
                secrets_path="rasa-secrets2",
                transit_mount_point="some transit key",
            ),
            MagicMock(),
        ),
    ],
)
def test_load_secret_manager(
    mock_read_secret_manager_config: Any,
    mock_create: MagicMock,
    config: SecretManagerConfig,
    secrets_manager: SecretsManager,
    monkeypatch: MonkeyPatch,
) -> None:
    mock_read_secret_manager_config.return_value = config

    mock_name = MagicMock()
    mock_name.return_value = config.secret_manager_type
    monkeypatch.setattr(secrets_manager, "name", mock_name)

    mock_create.return_value = secrets_manager

    load_secret_manager("some_file")
    provider = SecretsManagerProvider()

    assert provider.get_secrets_manager(config.secret_manager_type) == secrets_manager


def test_read_vault_endpoint_config() -> None:
    assert read_vault_endpoint_config(None) is None
