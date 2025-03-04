from typing import Dict, List, Union

import pytest

from rasa.shared.constants import AZURE_API_TYPE
from rasa.shared.providers._configs.azure_entra_id_config import (
    AZURE_AUTHORITY_FIELD,
    AZURE_CERTIFICATE_PASSWORD_FIELD,
    AZURE_CERTIFICATE_PATH_FIELD,
    AZURE_CLIENT_ID_FIELD,
    AZURE_CLIENT_SECRET_FIELD,
    AZURE_TENANT_ID_FIELD,
    AzureEntraIDOAuthConfig,
    AzureEntraIDOAuthType,
)
from rasa.shared.providers._configs.azure_openai_client_config import (
    AzureOpenAIClientConfig,
    OAuthConfigWrapper,
)
from rasa.shared.providers._configs.oauth_config import OAUTH_TYPE_FIELD


@pytest.mark.parametrize(
    "scopes",
    [
        ["scope1", "scope2"],
        "scope1",
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET.value,
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE.value,  # noqa: E501
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT.value,
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
    ],
)
def test_oauth_config_wrapper_from_config(
    config: Dict[str, str], scopes: Union[str, List[str]]
) -> None:
    """Tests that the from_config method returns the correct instance of the OAuthConfigWrapper."""  # noqa: E501

    oauth_config = {**config, "scopes": scopes}

    # We need to deepcopy the config as from_config methods modify the input
    oauth_result = AzureEntraIDOAuthConfig.from_dict(oauth_config)
    result = OAuthConfigWrapper.from_dict(oauth_config)

    # to keep the type checker happy
    assert isinstance(result, OAuthConfigWrapper)
    assert result.oauth == oauth_result
    assert result.original_config == oauth_config


@pytest.mark.parametrize(
    "scopes",
    [
        ["scope1", "scope2"],
        "scope1",
    ],
)
@pytest.mark.parametrize(
    "config",
    [
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET.value,
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE.value,  # noqa: E501
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
            AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
        {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT.value,
            AZURE_AUTHORITY_FIELD: "authority_host_value",
        },
    ],
)
def test_oauth_config_wrapper_to_dict(
    config: Dict[str, str], scopes: Union[str, List[str]]
) -> None:
    oauth_config = {**config, "scopes": scopes}
    """Tests that the to_dict method returns the original config."""

    # We need to deepcopy the config as from_dict methods modify the input
    result = OAuthConfigWrapper.from_dict(oauth_config)

    # to keep the type checker happy
    assert isinstance(result, OAuthConfigWrapper)
    assert result.to_dict() == oauth_config


@pytest.mark.parametrize(
    "general_config",
    [
        {
            "deployment": "deployment_value",
            "model": "model_value",
            "api_base": "api_base_value",
            "api_version": "api_version_value",
            "provider": "azure",
        }
    ],
)
@pytest.mark.parametrize(
    "oauth_scopes",
    [
        ["scope1", "scope2"],
        "scope1",
    ],
)
@pytest.mark.parametrize(
    "oauth_config_per_type",
    [
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET.value,  # noqa: E501
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE.value,  # noqa: E501
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
                AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT.value,
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
    ],
)
def test_azure_open_ai_client_config_oauth_from_config(
    general_config: Dict[str, str],
    oauth_scopes: Union[str, List[str]],
    oauth_config_per_type: Dict[str, str],
) -> None:
    """Tests that the AzureOpenAIClientConfig can be created from oauth config dict."""
    config = {
        **general_config,
        "oauth": {**oauth_config_per_type, "scopes": oauth_scopes},
    }

    result = AzureOpenAIClientConfig.from_dict(config)

    assert result.deployment == general_config["deployment"]
    assert result.model == general_config["model"]
    assert result.api_base == general_config["api_base"]
    assert result.api_version == general_config["api_version"]
    assert result.api_type == AZURE_API_TYPE
    assert result.provider == general_config["provider"]
    assert result.oauth == OAuthConfigWrapper.from_dict(
        {**oauth_config_per_type, "scopes": oauth_scopes}
    )


def test_azure_open_ai_client_config_from_config_api_key_and_oauth() -> None:
    """Tests that an error is raised when both api_key and oauth are provided."""

    config = {
        "deployment": "deployment_value",
        "model": "model_value",
        "api_base": "api_base_value",
        "api_version": "api_version_value",
        "provider": "azure",
        "api_key": "api_key_value",
        "oauth": {
            OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET.value,
            AZURE_CLIENT_ID_FIELD: "client_id_value",
            AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
            AZURE_TENANT_ID_FIELD: "tenant_id_value",
            AZURE_AUTHORITY_FIELD: "authority_host_value",
            "scopes": ["scope1", "scope2"],
        },
    }

    with pytest.raises(ValueError):
        AzureOpenAIClientConfig.from_dict(config)


@pytest.mark.parametrize(
    "general_config",
    [
        {
            "deployment": "deployment_value",
            "model": "model_value",
            "api_base": "api_base_value",
            "api_version": "api_version_value",
            "provider": "azure",
        }
    ],
)
@pytest.mark.parametrize(
    "oauth_scopes",
    [
        ["scope1", "scope2"],
        "scope1",
    ],
)
@pytest.mark.parametrize(
    "oauth_config_per_type",
    [
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET.value,  # noqa: E501
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_CLIENT_SECRET_FIELD: "client_secret_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE.value,  # noqa: E501
                AZURE_CLIENT_ID_FIELD: "client_id_value",
                AZURE_TENANT_ID_FIELD: "tenant_id_value",
                AZURE_CERTIFICATE_PATH_FIELD: "certificate_path_value",
                AZURE_CERTIFICATE_PASSWORD_FIELD: "certificate_password_value",
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
        (
            {
                OAUTH_TYPE_FIELD: AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT.value,
                AZURE_AUTHORITY_FIELD: "authority_host_value",
            }
        ),
    ],
)
def test_azure_open_ai_client_config_oauth_to_dict(
    general_config: Dict[str, str],
    oauth_scopes: Union[str, List[str]],
    oauth_config_per_type: Dict[str, str],
) -> None:
    """Tests that the AzureOpenAIClientConfig can be created from a oauth config dict."""  # noqa: E501
    config = {
        **general_config,
        "oauth": {**oauth_config_per_type, "scopes": oauth_scopes},
    }

    azure_config = AzureOpenAIClientConfig.from_dict(config)
    result = azure_config.to_dict()

    expected_result = {
        **config,
        "api_type": AZURE_API_TYPE,
    }
    assert result == expected_result
