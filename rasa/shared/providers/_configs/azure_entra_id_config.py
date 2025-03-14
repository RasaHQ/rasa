from __future__ import annotations

import abc
import logging
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Type

import structlog
from azure.core.credentials import TokenCredential
from azure.identity import (
    CertificateCredential,
    ClientSecretCredential,
    DefaultAzureCredential,
)
from pydantic import BaseModel, Field, SecretStr

from rasa.shared.providers._configs.oauth_config import OAUTH_TYPE_FIELD, OAuth

AZURE_CLIENT_ID_FIELD = "client_id"
AZURE_CLIENT_SECRET_FIELD = "client_secret"
AZURE_TENANT_ID_FIELD = "tenant_id"
AZURE_CERTIFICATE_PATH_FIELD = "certificate_path"
AZURE_CERTIFICATE_PASSWORD_FIELD = "certificate_password"
AZURE_SEND_CERTIFICATE_CHAIN_FIELD = "send_certificate_chain"
AZURE_SCOPES_FIELD = "scopes"
AZURE_AUTHORITY_FIELD = "authority_host"
AZURE_DISABLE_INSTANCE_DISCOVERY_FIELD = "disable_instance_discovery"


azure_logger = logging.getLogger("azure")
azure_logger.setLevel(logging.DEBUG)

structlogger = structlog.get_logger()


class AzureEntraIDOAuthType(str, Enum):
    """Azure Entra ID OAuth types."""

    AZURE_ENTRA_ID_DEFAULT = "azure_entra_id_default"
    AZURE_ENTRA_ID_CLIENT_SECRET = "azure_entra_id_client_secret"
    AZURE_ENTRA_ID_CLIENT_CERTIFICATE = "azure_entra_id_client_certificate"

    # Invalid type is used to indicate that the type
    # configuration is invalid EntraID or not set.
    INVALID = "invalid"

    @staticmethod
    def from_string(value: Optional[str]) -> AzureEntraIDOAuthType:
        """Converts a string to an AzureOAuthType."""
        if value is None or value not in AzureEntraIDOAuthType.valid_string_values():
            return AzureEntraIDOAuthType.INVALID

        return AzureEntraIDOAuthType(value)

    @staticmethod
    def valid_string_values() -> Set[str]:
        """Returns the valid string values for the AzureOAuthType."""
        return {e.value for e in AzureEntraIDOAuthType.valid_values()}

    @staticmethod
    def valid_values() -> Set[AzureEntraIDOAuthType]:
        """Returns the valid values for the AzureOAuthType."""
        return {
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT,
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET,
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE,
        }


# BearerTokenProvider is a callable that returns a bearer token.
BearerTokenProvider = Callable[[], str]


class AzureEntraIDTokenProviderConfig(abc.ABC):
    """Interface for Azure Entra ID OAuth credential configuration."""

    @abc.abstractmethod
    def create_azure_token_provider(self) -> TokenCredential:
        """Create an Azure Entra ID token provider."""
        ...

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls: Type[AzureEntraIDTokenProviderConfig], config: Dict[str, Any]
    ) -> AzureEntraIDTokenProviderConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Returns:
            AzureEntraIDCredential
        """
        ...


class AzureEntraIDClientCredentialsConfig(AzureEntraIDTokenProviderConfig, BaseModel):
    """Azure Entra ID OAuth client credentials configuration.

    Attributes:
        client_id: The client ID.
        client_secret: The client secret.
        tenant_id: The tenant ID.
        authority_host: The authority host.
        disable_instance_discovery: Whether to disable instance discovery. This is used
            to disable fetching metadata from the Azure Instance Metadata Service.
    """

    client_id: str = Field(min_length=1)
    client_secret: SecretStr = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    authority_host: Optional[str] = None
    disable_instance_discovery: bool = False

    @staticmethod
    def required_fields() -> Set[str]:
        """Returns the required fields for the configuration."""
        return {AZURE_CLIENT_ID_FIELD, AZURE_TENANT_ID_FIELD, AZURE_CLIENT_SECRET_FIELD}

    @staticmethod
    def config_has_required_fields(config: Dict[str, Any]) -> bool:
        """Check if the configuration has all the required fields."""
        return AzureEntraIDClientCredentialsConfig.required_fields().issubset(
            set(config.keys())
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> AzureEntraIDClientCredentialsConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Returns:
            AzureClientCredentialsConfig
        """
        if not cls.config_has_required_fields(config):
            message = (
                f"A configuration for Azure client credentials "
                f"must contain the following keys: {cls.required_fields()}"
            )
            structlogger.error(
                "azure_client_credentials_config.missing_required_keys",
                message=message,
                config=config,
            )
            raise ValueError(message)

        return cls(
            client_id=config.pop(AZURE_CLIENT_ID_FIELD),
            client_secret=config.pop(AZURE_CLIENT_SECRET_FIELD),
            tenant_id=config.pop(AZURE_TENANT_ID_FIELD),
            authority_host=config.pop(AZURE_AUTHORITY_FIELD, None),
            disable_instance_discovery=config.pop(
                AZURE_DISABLE_INSTANCE_DISCOVERY_FIELD, False
            ),
        )

    def create_azure_token_provider(self) -> TokenCredential:
        """Create a ClientSecretCredential for Azure Entra ID."""
        return create_azure_entra_id_client_credentials(
            client_id=self.client_id,
            client_secret=self.client_secret.get_secret_value(),
            tenant_id=self.tenant_id,
            authority_host=self.authority_host,
            disable_instance_discovery=self.disable_instance_discovery,
        )


# We are caching the result of this function to preserve the refresh
# token which is stored inside the credential object.
# This allows us to reuse the same credential object (refresh token)
# across multiple requests.
# Refresh token is used to get a new access token when the current access
# token expires without having to re-authenticate the
# user (transmit the client secret again).
@lru_cache
def create_azure_entra_id_client_credentials(
    client_id: str,
    client_secret: str,
    tenant_id: str,
    authority_host: Optional[str] = None,
    disable_instance_discovery: bool = False,
) -> ClientSecretCredential:
    """Creates a ClientSecretCredential for Azure Entra ID.

    We cache the result of this function to avoid creating multiple instances
    of the same credential. This makes it possible to utilise the token caching
    and token refreshing functionality of the azure-identity library.

    Args:
        client_id: The client ID.
        client_secret: The client secret.
        tenant_id: The tenant ID.
        authority_host: The authority host.
        disable_instance_discovery: Whether to disable instance discovery. This is used
            to disable fetching metadata from the Azure Instance Metadata Service.

    Returns:
        ClientSecretCredential
    """
    return ClientSecretCredential(
        client_id=client_id,
        client_secret=client_secret,
        tenant_id=tenant_id,
        authority=authority_host,
        disable_instance_discovery=disable_instance_discovery,
    )


class AzureEntraIDClientCertificateConfig(AzureEntraIDTokenProviderConfig, BaseModel):
    """Azure Entra ID OAuth client certificate configuration.

    Attributes:
        client_id: The client ID.
        tenant_id: The tenant ID.
        certificate_path: The path to the certificate file.
        certificate_password: The certificate password.
        send_certificate_chain: Whether to send the certificate chain.
        authority_host: The authority host.
        disable_instance_discovery: Whether to disable instance discovery. This is used
            to disable fetching metadata from the Azure Instance Metadata Service.
    """

    client_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    certificate_path: str = Field(min_length=1)
    certificate_password: Optional[SecretStr] = None
    send_certificate_chain: bool = False
    authority_host: Optional[str] = None
    disable_instance_discovery: bool = False

    @staticmethod
    def required_fields() -> Set[str]:
        """Returns the required fields for the configuration."""
        return {
            AZURE_CLIENT_ID_FIELD,
            AZURE_TENANT_ID_FIELD,
            AZURE_CERTIFICATE_PATH_FIELD,
        }

    @staticmethod
    def config_has_required_fields(config: Dict[str, Any]) -> bool:
        """Check if the configuration has all the required fields."""
        return AzureEntraIDClientCertificateConfig.required_fields().issubset(
            set(config.keys())
        )

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> AzureEntraIDClientCertificateConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Returns:
            AzureClientCertificateConfig
        """
        if not cls.config_has_required_fields(config):
            message = (
                f"A configuration for Azure client certificate "
                f"must contain "
                f"the following keys: {cls.required_fields()}"
            )
            structlogger.error(
                "azure_client_certificate_config.validation_error",
                message=message,
                config=config,
            )
            raise ValueError(message)

        return cls(
            client_id=config[AZURE_CLIENT_ID_FIELD],
            tenant_id=config[AZURE_TENANT_ID_FIELD],
            certificate_path=config[AZURE_CERTIFICATE_PATH_FIELD],
            certificate_password=config.get(AZURE_CERTIFICATE_PASSWORD_FIELD, None),
            authority_host=config.get(AZURE_AUTHORITY_FIELD, None),
            send_certificate_chain=config.get(
                AZURE_SEND_CERTIFICATE_CHAIN_FIELD, False
            ),
            disable_instance_discovery=config.get(
                AZURE_DISABLE_INSTANCE_DISCOVERY_FIELD, False
            ),
        )

    def create_azure_token_provider(self) -> TokenCredential:
        """Creates a CertificateCredential for Azure Entra ID."""
        return create_azure_entra_id_certificate_credentials(
            client_id=self.client_id,
            tenant_id=self.tenant_id,
            certificate_path=self.certificate_path,
            password=self.certificate_password.get_secret_value()
            if self.certificate_password
            else None,
            send_certificate_chain=self.send_certificate_chain,
            authority_host=self.authority_host,
            disable_instance_discovery=self.disable_instance_discovery,
        )


# We are caching the result of this function to preserve the refresh
# token which is stored inside the credential object.
# This allows us to reuse the same credential object (refresh token)
# across multiple requests.
# Refresh token is used to get a new access token when the current
# access token expires without having to re-authenticate
# the user (transmit the client certificate again).
@lru_cache
def create_azure_entra_id_certificate_credentials(
    tenant_id: str,
    client_id: str,
    certificate_path: Optional[str] = None,
    password: Optional[str] = None,
    send_certificate_chain: bool = False,
    authority_host: Optional[str] = None,
    disable_instance_discovery: bool = False,
) -> CertificateCredential:
    """Creates a CertificateCredential for Azure Entra ID.

    We cache the result of this function to avoid creating multiple instances
    of the same credential. This makes it possible to utilise the token caching
    and token refreshing functionality of the azure-identity library.

    Args:
        tenant_id: The tenant ID.
        client_id: The client ID.
        certificate_path: The path to the certificate file.
        password: The certificate password.
        send_certificate_chain: Whether to send the certificate chain.
        authority_host: The authority host.
        disable_instance_discovery: Whether to disable instance discovery. This is used

    Returns:
        CertificateCredential
    """

    return CertificateCredential(
        client_id=client_id,
        tenant_id=tenant_id,
        certificate_path=certificate_path,
        password=password.encode("utf-8") if password else None,
        send_certificate_chain=send_certificate_chain,
        authority=authority_host,
        disable_instance_discovery=disable_instance_discovery,
    )


class AzureEntraIDDefaultCredentialsConfig(AzureEntraIDTokenProviderConfig, BaseModel):
    """Azure Entra ID OAuth default credentials configuration.

    Attributes:
        authority_host: The authority host.
    """

    authority_host: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> AzureEntraIDDefaultCredentialsConfig:
        """Initializes a dataclass from the passed config.

        Args:
            config: (dict) The config from which to initialize.

        Returns:
            AzureOAuthDefaultCredentialsConfig
        """
        return cls(authority_host=config.pop(AZURE_AUTHORITY_FIELD, None))

    def create_azure_token_provider(self) -> TokenCredential:
        """Creates a DefaultAzureCredential."""
        return create_azure_entra_id_default_credentials(
            authority_host=self.authority_host
        )


@lru_cache
def create_azure_entra_id_default_credentials(
    authority_host: Optional[str] = None,
) -> DefaultAzureCredential:
    """Creates a DefaultAzureCredential.

    We cache the result of this function to avoid creating multiple instances
    of the same credential. This makes it possible to utilise the token caching
    functionality of the azure-identity library.

    Args:
        authority_host: The authority host.

    Returns:
        DefaultAzureCredential
    """
    return DefaultAzureCredential(authority=authority_host)


class AzureEntraIDOAuthConfig(OAuth, BaseModel):
    """Azure Entra ID OAuth configuration.

    It consists of the scopes and the Azure Entra ID OAuth credentials.
    """

    # pydantic configuration to allow arbitrary user defined types
    class Config:
        arbitrary_types_allowed = True

    scopes: List[str]
    azure_entra_id_token_provider_config: AzureEntraIDTokenProviderConfig

    @staticmethod
    def _supported_azure_oauth() -> (
        Dict[AzureEntraIDOAuthType, Type[AzureEntraIDTokenProviderConfig]]
    ):
        """Returns a mapping of supported Azure Entra ID OAuth types to their"""
        return {
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_DEFAULT: AzureEntraIDDefaultCredentialsConfig,  # noqa: E501
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_SECRET: AzureEntraIDClientCredentialsConfig,  # noqa: E501
            AzureEntraIDOAuthType.AZURE_ENTRA_ID_CLIENT_CERTIFICATE: AzureEntraIDClientCertificateConfig,  # noqa: E501
        }

    @staticmethod
    def _get_azure_oauth_by_type(
        oauth_type: AzureEntraIDOAuthType,
    ) -> Type[AzureEntraIDTokenProviderConfig]:
        """Returns the Azure Entra ID OAuth class based on the type.

        Args:
            oauth_type: (AzureOAuthType) The type of the Azure Entra ID OAuth.

        Returns:
            The Azure Entra ID OAuth class

        Raises:
            ValueError: If the passed oauth_type is not supported or invalid.
        """
        azure_oauth_types = AzureEntraIDOAuthConfig._supported_azure_oauth()
        azure_oauth_class = azure_oauth_types.get(oauth_type)

        if azure_oauth_class is None:
            message = (
                f"Unsupported Azure Entra ID oauth type: {oauth_type}. "
                f"Supported types are: {AzureEntraIDOAuthType.valid_string_values()}"
            )
            structlogger.error(
                "azure_oauth_config.unsupported_azure_oauth_type",
                message=message,
            )
            raise ValueError(message)

        return azure_oauth_class

    @classmethod
    def from_dict(cls, oauth_config: Dict[str, Any]) -> AzureEntraIDOAuthConfig:
        """Initializes a dataclass from the passed config.

        Args:
            oauth_config: (dict) The config from which to initialize.

        Returns:
            AzureOAuthConfig
        """

        config = deepcopy(oauth_config)

        scopes = AzureEntraIDOAuthConfig._read_scopes_from_config(config)
        azure_credentials = (
            AzureEntraIDOAuthConfig._create_azure_entra_id_client_from_config(config)
        )
        return cls(
            azure_entra_id_token_provider_config=azure_credentials, scopes=scopes
        )

    @staticmethod
    def _read_scopes_from_config(oauth_config: Dict[str, Any]) -> List[str]:
        """Reads scopes from the configuration.

        The original scopes are removed from the configuration.

        Args:
            oauth_config: (dict) The configuration from which to read the scopes.

        Returns:
            List[str]: The list of scopes.
        """
        scopes = oauth_config.pop(AZURE_SCOPES_FIELD, "")

        if not scopes:
            message = "Azure Entra ID scopes cannot be empty."
            structlogger.error(
                "azure_oauth_config.scopes_empty",
                message=message,
            )
            raise ValueError(message)

        if isinstance(scopes, str):
            scopes = [scopes]

        return scopes

    @staticmethod
    def _create_azure_entra_id_client_from_config(
        oauth_config: Dict[str, Any],
    ) -> AzureEntraIDTokenProviderConfig:
        """Creates an Azure Entra ID client from the configuration.

        Args:
            oauth_config: (dict) The configuration from which to create the credential.

        Returns:
            AzureEntraIDTokenProviderConfig: The Azure OAuth credential.
        """

        oauth_type = AzureEntraIDOAuthType.from_string(
            oauth_config.pop(OAUTH_TYPE_FIELD, None)
        )

        if oauth_type == AzureEntraIDOAuthType.INVALID:
            message = (
                "Azure Entra ID oauth configuration must contain "
                f"'{OAUTH_TYPE_FIELD}' field and it must be set to one of the "
                f"following values: {AzureEntraIDOAuthType.valid_string_values()}, "
            )
            structlogger.error(
                "azure_oauth_config.missing_azure_oauth_type",
                message=message,
            )
            raise ValueError(message)

        azure_oauth_class = AzureEntraIDOAuthConfig._get_azure_oauth_by_type(oauth_type)
        return azure_oauth_class.from_dict(oauth_config)

    def create_azure_credential(
        self,
    ) -> TokenCredential:
        """Create an Azure Entra ID client which can be used to get a bearer token."""
        return self.azure_entra_id_token_provider_config.create_azure_token_provider()

    def get_bearer_token(self) -> str:
        """Returns a bearer token."""
        return self.create_azure_credential().get_token(*self.scopes).token
