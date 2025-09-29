from __future__ import annotations

import os
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

import structlog
from pydantic import BaseModel

from rasa.core.constants import IAM_CLOUD_PROVIDER_ENV_VAR_NAME

structlogger = structlog.get_logger(__name__)


class TemporaryCredentials(BaseModel):
    """Dataclass storing temporary credentials."""

    auth_token: Optional[str] = None
    expiration: Optional[float] = None
    username: Optional[str] = None
    presigned_url: Optional[str] = None


@runtime_checkable
class IAMCredentialsProvider(Protocol):
    """Interface for generating temporary credentials using IAM roles."""

    def get_temporary_credentials(self) -> TemporaryCredentials:
        """Generates temporary credentials using IAM roles."""
        ...


class IAMCredentialsProviderType(Enum):
    """Enum for supported IAM credentials provider types."""

    AWS = "aws"


class SupportedServiceType(Enum):
    """Enum for supported services using IAM credentials providers."""

    TRACKER_STORE = "tracker_store"
    EVENT_BROKER = "event_broker"
    LOCK_STORE = "lock_store"


class IAMCredentialsProviderInput(BaseModel):
    """Input data for creating an IAM credentials provider."""

    service_type: SupportedServiceType
    service_name: str
    username: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    cluster_name: Optional[str] = None


def create_iam_credentials_provider(
    provider_input: IAMCredentialsProviderInput,
) -> Optional[IAMCredentialsProvider]:
    """Factory function to create an IAM credentials provider.

    Args:
        provider_input: Input data for creating an IAM credentials provider.

    Returns:
        An instance of the specified IAM credentials provider or
        None if the type is unsupported.
    """
    iam_cloud_provider = os.getenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME)

    if iam_cloud_provider is None:
        return None

    try:
        provider_type = IAMCredentialsProviderType(iam_cloud_provider.lower())
    except ValueError:
        structlogger.warning(
            "rasa.core.iam_credentials_provider.create_iam_credentials_provider.unsupported_provider",
            event_info=f"Unsupported IAM cloud provider: {iam_cloud_provider}",
        )
        return None

    if provider_type == IAMCredentialsProviderType.AWS:
        from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
            create_aws_iam_credentials_provider,
        )

        return create_aws_iam_credentials_provider(provider_input)

    return None
