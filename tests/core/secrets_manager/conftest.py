from typing import Any, Callable, Dict, Optional, Text, Tuple
from unittest.mock import MagicMock

import pytest
from _pytest.monkeypatch import MonkeyPatch

from rasa.core.secrets_manager.secret_manager import EndpointResolver
from rasa.core.secrets_manager.vault import VaultSecretsManager, VaultTokenManager


@pytest.fixture
def vault_secrets_manager(
    credentials: Tuple[Text, Text],
    monkeypatch: MonkeyPatch,
) -> VaultSecretsManager:
    def mock_start(_: Any) -> None:
        pass

    monkeypatch.setattr(VaultTokenManager, "start", mock_start)

    mock_load_secrets = MagicMock()
    mock_load_secrets.return_value = {
        "username": credentials[0],
        "password": credentials[1],
    }

    monkeypatch.setattr(VaultSecretsManager, "load_secrets", mock_load_secrets)

    return VaultSecretsManager(
        # deepcode ignore HardcodedNonCryptoSecret/test: Test secret
        host="localhost:8200",
        token="myroot",
        # deepcode ignore HardcodedNonCryptoSecret/test: Test credential
        secrets_path="rasa-secrets",
    )


@pytest.fixture
def mock_get_credentials_from_secret_managers() -> MagicMock:
    """Return Mock for `_get_credentials_from_secret_managers`."""
    return MagicMock()


@pytest.fixture
def make_mock_get_credentials_from_secret_managers(
    mock_get_credentials_from_secret_managers: MagicMock,
) -> Callable[[Optional[Dict[Text, Text]]], MagicMock]:
    """Create factory which outputs a mock for `_get_credentials_from_secret_managers`.

    Returns:
        A function which creates a mock for `_get_credentials_from_secret_managers`.
    """

    def _make_mock_get_credentials_from_secret_managers(
        result_credentials: Optional[Dict[Text, Text]],
    ) -> MagicMock:
        mock_get_credentials_from_secret_managers.return_value = result_credentials
        return mock_get_credentials_from_secret_managers

    return _make_mock_get_credentials_from_secret_managers


@pytest.fixture
def mock_get_secret_manager() -> MagicMock:
    """Return Mock for `_get_secret_manager`."""
    return MagicMock()


@pytest.fixture
def mock_endpoint_resolver_get_secret_manager(
    mock_get_secret_manager: MagicMock,
    vault_secrets_manager: VaultSecretsManager,
    monkeypatch: MonkeyPatch,
) -> Callable[[], None]:
    """Create factory which outputs a mock for `EndpointResolver._get_secret_manager`.

    Returns:
        A function which sets a mock for `EndpointResolver._get_secret_manager`.
    """

    def _mock_endpoint_resolver_get_secret_manager() -> None:
        mock_get_secret_manager.return_value = vault_secrets_manager
        monkeypatch.setattr(
            EndpointResolver, "_get_secret_manager", mock_get_secret_manager
        )

    return _mock_endpoint_resolver_get_secret_manager
