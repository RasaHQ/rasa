# file deepcode ignore HardcodedNonCryptoSecret/test: Secrets are all just examples for tests. # noqa: E501

from typing import Callable, Dict, Optional, Text, Tuple
from unittest.mock import MagicMock

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from rasa.utils.endpoints import EndpointConfig

from rasa.core.secrets_manager.secret_manager import (
    EndpointResolver,
    SecretsManagerProvider,
)
from rasa.core.secrets_manager.vault import VaultSecretsManager


@pytest.fixture
def credentials() -> Tuple[Text, Text]:
    return "myusername", "mypassword"


def test_endpoint_resolver_update_credentials_from_secret_manager(
    vault_secrets_manager: VaultSecretsManager,
    credentials: Tuple[Text, Text],
    monkeypatch: MonkeyPatch,
) -> None:
    mock_get_secret_manager = MagicMock()

    mock_load_secrets = MagicMock()
    mock_load_secrets.return_value = {
        "sql_store_username": credentials[0],
        "sql_store_password": credentials[1],
    }
    monkeypatch.setattr(vault_secrets_manager, "load_secrets", mock_load_secrets)

    mock_get_secret_manager.return_value = vault_secrets_manager
    monkeypatch.setattr(
        EndpointResolver, "_get_secret_manager", mock_get_secret_manager
    )

    secret_manager_name = "vault"
    result = EndpointResolver._get_credentials_from_secret_manager(
        secret_manager_name=secret_manager_name,
        endpoint_config=EndpointConfig(
            username={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_username",
            },
            password={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_password",
            },
        ),
        endpoint_properties=["username", "password"],
    )

    assert result is not None
    assert len(result) == 2
    assert result["username"] == credentials[0]
    assert result["password"] == credentials[1]


def test_endpoint_resolver_update_credentials_from_secret_manager_secret_not_found(
    caplog: LogCaptureFixture,
    vault_secrets_manager: VaultSecretsManager,
    monkeypatch: MonkeyPatch,
    mock_endpoint_resolver_get_secret_manager: Callable[[], None],
) -> None:
    mock_load_secrets = MagicMock()
    mock_load_secrets.return_value = {
        "sql_store_url": "some value",
    }
    monkeypatch.setattr(vault_secrets_manager, "load_secrets", mock_load_secrets)

    endpoint_properties = ["username", "password"]
    mock_get_secret_manager = MagicMock()
    mock_get_secret_manager.return_value = vault_secrets_manager
    monkeypatch.setattr(
        SecretsManagerProvider, "get_secrets_manager", mock_get_secret_manager
    )

    secret_manager_name = "vault"
    result = EndpointResolver._get_credentials_from_secret_manager(
        secret_manager_name=secret_manager_name,
        endpoint_config=EndpointConfig(
            username={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_username",
            },
            password={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_password",
            },
        ),
        endpoint_properties=endpoint_properties,
    )

    assert result is None

    for endpoint_property_name in endpoint_properties:
        log_msg = f"Failed to load {endpoint_property_name} from secrets manager "
        assert log_msg in caplog.text


def test_endpoint_resolver_update_credentials_from_secret_manager_no_secrets(
    caplog: LogCaptureFixture,
    vault_secrets_manager: VaultSecretsManager,
    monkeypatch: MonkeyPatch,
    mock_endpoint_resolver_get_secret_manager: Callable[[], None],
) -> None:
    monkeypatch.setattr(
        vault_secrets_manager, "load_secrets", MagicMock(return_value={})
    )

    mock_get_secret_manager = MagicMock()
    mock_get_secret_manager.return_value = vault_secrets_manager
    monkeypatch.setattr(
        SecretsManagerProvider, "get_secrets_manager", mock_get_secret_manager
    )

    result = EndpointResolver._get_credentials_from_secret_manager(
        secret_manager_name="vault",
        endpoint_config=EndpointConfig(
            username={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_username",
            },
            password={
                "source": "secrets_manager.vault",
                "secret_key": "sql_store_password",
            },
        ),
        endpoint_properties=["username", "password"],
    )

    assert result is None


@pytest.mark.parametrize(
    "endpoint_config, result_credentials, expected_endpoint_config",
    [
        (
            EndpointConfig(
                username="username",
                password="password",
            ),
            {
                "username": "new username",
                # deepcode ignore NoHardcodedPasswords/test: Test credential
                "password": "new password",
            },
            EndpointConfig(
                username="new username",
                password="new password",
            ),
        ),
    ],
)
def test_endpoint_resolver_update_config(
    endpoint_config: EndpointConfig,
    result_credentials: Optional[Dict[Text, Text]],
    expected_endpoint_config: EndpointConfig,
    monkeypatch: MonkeyPatch,
    make_mock_get_credentials_from_secret_managers: MagicMock,
) -> None:
    mock_get_credentials_from_secret_managers = (
        make_mock_get_credentials_from_secret_managers(result_credentials)
    )
    monkeypatch.setattr(
        EndpointResolver,
        "_get_credentials_from_secret_managers",
        mock_get_credentials_from_secret_managers,
    )

    result = EndpointResolver.update_config(endpoint_config)

    assert isinstance(result, EndpointConfig)
    assert result.kwargs == expected_endpoint_config.kwargs


@pytest.mark.parametrize(
    "endpoint_config, result_credentials, expected_endpoint_config",
    [
        (
            EndpointConfig(
                username="username",
                password="password",
            ),
            None,
            EndpointConfig(
                username="username",
                password="password",
            ),
        ),
    ],
)
def test_endpoint_resolver_update_config_no_secrets_from_secret_manager(
    endpoint_config: EndpointConfig,
    result_credentials: Optional[Dict[Text, Text]],
    expected_endpoint_config: EndpointConfig,
    monkeypatch: MonkeyPatch,
    make_mock_get_credentials_from_secret_managers: MagicMock,
) -> None:
    mock_get_credentials_from_secret_managers = (
        make_mock_get_credentials_from_secret_managers(result_credentials)
    )
    monkeypatch.setattr(
        EndpointResolver,
        "_get_credentials_from_secret_managers",
        mock_get_credentials_from_secret_managers,
    )

    result = EndpointResolver.update_config(endpoint_config)

    assert isinstance(result, EndpointConfig)
    assert result.kwargs == expected_endpoint_config.kwargs


@pytest.mark.parametrize(
    "endpoint_config, endpoint_property_name, expected_result",
    [
        (
            EndpointConfig(
                username="username",
                password="password",
            ),
            "non_existing_property",
            None,
        ),
        (
            EndpointConfig(
                username="username",
                password="password",
            ),
            "username",
            None,
        ),
        (
            EndpointConfig(
                username={
                    "source": "secrets_manager.vault",
                    "secret_key": "sql_store_username",
                },
                password="password",
            ),
            "username",
            "sql_store_username",
        ),
    ],
)
def test_auth_retry_tracker_get_secret_name(
    endpoint_config: EndpointConfig,
    endpoint_property_name: Text,
    expected_result: Optional[Text],
) -> None:
    assert (
        EndpointResolver.get_secret_name(
            endpoint_config=endpoint_config,
            endpoint_property_name=endpoint_property_name,
        )
        == expected_result
    )
