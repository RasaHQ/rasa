from copy import deepcopy
from typing import Any, Dict

import pytest

from rasa.shared.providers._configs.azure_entra_id_config import (
    AZURE_AUTHORITY_FIELD,
    AZURE_CERTIFICATE_PASSWORD_FIELD,
    AZURE_CERTIFICATE_PATH_FIELD,
    AZURE_CLIENT_ID_FIELD,
    AZURE_CLIENT_SECRET_FIELD,
    AZURE_SCOPES_FIELD,
    AZURE_TENANT_ID_FIELD,
    AzureEntraIDClientCertificateConfig,
    AzureEntraIDClientCredentialsConfig,
    AzureEntraIDDefaultCredentialsConfig,
    AzureEntraIDOAuthConfig,
    AzureEntraIDOAuthType,
    AzureEntraIDTokenProviderConfig,
)
from rasa.shared.providers._configs.oauth_config import OAUTH_TYPE_FIELD


def test_azure_entra_id_client_credentials_required_fields() -> None:
    assert AzureEntraIDClientCredentialsConfig.required_fields().issubset(
        {AZURE_CLIENT_ID_FIELD, AZURE_TENANT_ID_FIELD, AZURE_CLIENT_SECRET_FIELD}
    )

    assert AzureEntraIDClientCredentialsConfig.required_fields().issuperset(
        {AZURE_CLIENT_ID_FIELD, AZURE_TENANT_ID_FIELD, AZURE_CLIENT_SECRET_FIELD}
    )


@pytest.mark.parametrize(
    "config, expected_value",
    [
        ({}, False),
        ({AZURE_CLIENT_ID_FIELD: "client_id_value"}, False),
        ({AZURE_TENANT_ID_FIELD: "tenant_id_value"}, False),
        ({AZURE_CLIENT_SECRET_FIELD: "client_secret_value"}, False),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
            },
            False,
        ),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            },
            False,
        ),
        (
            {
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            },
            False,
        ),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            },
            True,
        ),
    ],
)
def test_azure_entra_id_client_credentials_config_has_required_fields(
    config: dict, expected_value: bool
) -> None:
    assert (
        AzureEntraIDClientCredentialsConfig.config_has_required_fields(config)
        == expected_value
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {AZURE_CLIENT_ID_FIELD: "client_id_value"},
        {AZURE_TENANT_ID_FIELD: "tenant_id_value"},
        {AZURE_CLIENT_SECRET_FIELD: "client_secret_value"},
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
        },
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
        },
        {
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
        },
    ],
)
def test_azure_entra_id_client_credentials_from_config_missing_field(
    config: Dict[str, Any],
) -> None:
    """Tests that the method raises a ValueError when a required field is missing."""
    with pytest.raises(ValueError):
        AzureEntraIDClientCredentialsConfig.from_dict(config)


def test_azure_entra_id_client_credentials_from_config() -> None:
    config = {
        AZURE_CLIENT_ID_FIELD: "client_id_value",
        AZURE_TENANT_ID_FIELD: "tenant_id_value",
        AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
    }

    azure_entra_id_client_credentials_config = (
        AzureEntraIDClientCredentialsConfig.from_dict(config)
    )

    assert azure_entra_id_client_credentials_config.client_id == "client_id_value"
    assert azure_entra_id_client_credentials_config.tenant_id == "tenant_id_value"
    assert (
        azure_entra_id_client_credentials_config.client_secret.get_secret_value()
        == "client_secret_value"
    )


def test_azure_entra_id_client_certificate_required_fields() -> None:
    assert AzureEntraIDClientCertificateConfig.required_fields().issubset(
        {
            AZURE_CLIENT_ID_FIELD,
            AZURE_TENANT_ID_FIELD,
            AZURE_CERTIFICATE_PATH_FIELD,
        }
    )

    assert AzureEntraIDClientCertificateConfig.required_fields().issuperset(
        {
            AZURE_CLIENT_ID_FIELD,
            AZURE_TENANT_ID_FIELD,
            AZURE_CERTIFICATE_PATH_FIELD,
        }
    )


@pytest.mark.parametrize(
    "config, expected_value",
    [
        ({}, False),
        ({AZURE_CLIENT_ID_FIELD: "client_id_value"}, False),
        ({AZURE_TENANT_ID_FIELD: "tenant_id_value"}, False),
        ({AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value"}, False),
        ({AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value"}, False),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
            },
            False,
        ),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            },
            False,
        ),
        (
            {
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            },
            False,
        ),
        (
            {
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            },
            True,
        ),
    ],
)
def test_azure_entra_id_client_certificate_config_has_required_fields(
    config: dict, expected_value: bool
) -> None:
    assert (
        AzureEntraIDClientCertificateConfig.config_has_required_fields(config)
        == expected_value
    )


@pytest.mark.parametrize(
    "config",
    [
        {},
        {AZURE_CLIENT_ID_FIELD: "client_id_value"},
        {AZURE_TENANT_ID_FIELD: "tenant_id_value"},
        {AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value"},
        {AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value"},
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
        },
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
        },
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
        },
        {
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
        },
        {
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
        },
        {
            AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
        },
        {
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
        },
    ],
)
def test_azure_entra_id_client_certificate_from_config_missing_field(
    config: Dict[str, Any],
) -> None:
    """Tests that the method raises a ValueError when a required field is missing."""
    with pytest.raises(ValueError):
        AzureEntraIDClientCertificateConfig.from_dict(config)


def test_azure_entra_id_client_certificate_from_config() -> None:
    config = {
        AZURE_CLIENT_ID_FIELD: "client_id_value",
        AZURE_TENANT_ID_FIELD: "tenant_id_value",
        AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
        AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
    }

    azure_entra_id_client_certificate_config = (
        AzureEntraIDClientCertificateConfig.from_dict(config)
    )

    assert azure_entra_id_client_certificate_config.client_id == "client_id_value"
    assert azure_entra_id_client_certificate_config.tenant_id == "tenant_id_value"
    assert (
        azure_entra_id_client_certificate_config.certificate_path
        == "certificate_path_value"
    )
    assert (
        azure_entra_id_client_certificate_config.certificate_password.get_secret_value()
        == "certificate_password_value"
    )


@pytest.mark.parametrize(
    "config, expected_value",
    [
        ({}, None),
        ({AZURE_AUTHORITY_FIELD: "authority_value"}, "authority_value"),
    ],
)
def test_azure_entra_id_default_credentials_from_config(
    config: Dict[str, Any], expected_value: str
) -> None:
    result = AzureEntraIDDefaultCredentialsConfig.from_dict(config)
    assert result.authority_host == expected_value


@pytest.mark.parametrize(
    "scopes, expected_scopes",
    [
        (["scope1", "scope2"], ["scope1", "scope2"]),
        ("scope1", ["scope1"]),
    ],
)
@pytest.mark.parametrize(
    "config, expected_config",
    [
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET,
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            },
            AzureEntraIDClientCredentialsConfig(
                client_id="client_id_value",
                tenant_id="tenant_id_value",
                client_secret="client_secret_value",
                authority_host="authority_host_value",
            ),
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE,  # noqa: E501
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
                AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            },
            AzureEntraIDClientCertificateConfig(
                client_id="client_id_value",
                tenant_id="tenant_id_value",
                certificate_path="certificate_path_value",
                certificate_password="certificate_password_value",
                authority_host="authority_host_value",
            ),
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT,
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            },
            AzureEntraIDDefaultCredentialsConfig(authority_host="authority_host_value"),
        ),
    ],
)
def test_azure_entra_id_oauth_config_from_config(
    config: Dict[str, Any],
    expected_config: AzureEntraIDTokenProviderConfig,
    scopes: Any,
    expected_scopes: Any,
) -> None:
    oauth_config = {
        **config,
        AZURE_SCOPES_FIELD: scopes,
    }

    result = AzureEntraIDOAuthConfig.from_dict(deepcopy(oauth_config))

    expected_value = AzureEntraIDOAuthConfig(
        scopes=expected_scopes,
        azure_entra_id_token_provider_config=expected_config,
    )
    assert result == expected_value


def test_azure_entra_id_oauth_config_from_config_empty_scopes() -> None:
    config = {
        OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET,
        AZURE_CLIENT_ID_FIELD: "client_id_value",
        AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
        AZURE_TENANT_ID_FIELD: "tenant_id_value",
        AZURE_AUTHORITY_FIELD: "authority_host_value",
    }

    with pytest.raises(ValueError):
        AzureEntraIDOAuthConfig.from_dict(config)


def test_azure_entra_id_oauth_config_from_config_invalid_type() -> None:
    config = {
        OAUTH_TYPE_FIELD: "invalid_type",
        AZURE_CLIENT_ID_FIELD: "client_id_value",
        AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
        AZURE_TENANT_ID_FIELD: "tenant_id_value",
        AZURE_AUTHORITY_FIELD: "authority_host_value",
        AZURE_SCOPES_FIELD: ["scope1", "scope2"],
    }

    with pytest.raises(ValueError):
        AzureEntraIDOAuthConfig.from_dict(config)
