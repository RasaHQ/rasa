import os
import threading
import time
from typing import Optional

import boto3
import structlog
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from botocore.exceptions import BotoCoreError

from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
    TemporaryCredentials,
)
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger(__name__)


class AWSRDSIAMCredentialsProvider(IAMCredentialsProvider):
    """Generates temporary credentials for AWS RDS using IAM roles."""

    def __init__(self, username: str, host: str, port: int) -> None:
        """Initializes the provider."""
        self.username = username
        self.host = host
        self.port = port

    def get_credentials(self) -> TemporaryCredentials:
        """Generates temporary credentials for AWS RDS."""
        structlogger.debug(
            "rasa.core.aws_rds_iam_credentials_provider.get_credentials",
            event_info="IAM authentication for AWS RDS enabled. "
            "Generating temporary auth token...",
        )

        try:
            client = boto3.client("rds")
            auth_token = client.generate_db_auth_token(
                DBHostname=self.host,
                Port=self.port,
                DBUsername=self.username,
            )
            structlogger.info(
                "rasa.core.aws_rds_iam_credentials_provider.generated_credentials",
                event_info="Successfully generated temporary auth token for AWS RDS.",
            )
            return TemporaryCredentials(auth_token=auth_token)
        except (BotoCoreError, ValueError) as exc:
            structlogger.error(
                "rasa.core.aws_rds_iam_credentials_provider.error_generating_credentials",
                event_info="Failed to generate temporary auth token for AWS RDS.",
                error=str(exc),
            )
            return TemporaryCredentials(auth_token=None)


class AWSMSKafkaIAMCredentialsProvider(IAMCredentialsProvider):
    """Generates temporary credentials for AWS MSK using IAM roles."""

    def __init__(self) -> None:
        self.region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        self._token: Optional[str] = None
        self._expires_at: float = 0
        self.refresh_margin_seconds = 60  # Refresh 60 seconds before expiry
        # ensure thread safety when refreshing token because the
        # kafka client library we use (confluent-kafka) is multithreaded
        self.lock = threading.Lock()

    @property
    def token(self) -> Optional[str]:
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        self._token = value

    @property
    def expires_at(self) -> float:
        return self._expires_at

    @expires_at.setter
    def expires_at(self, value: float) -> None:
        self._expires_at = value

    def get_credentials(self) -> TemporaryCredentials:
        """Generates temporary credentials for AWS MSK."""
        with self.lock:
            current_time = time.time()  # Current time in seconds
            if (
                not self.token
                or current_time >= self.expires_at - self.refresh_margin_seconds
            ):
                try:
                    auth_token, expiry_ms = MSKAuthTokenProvider.generate_auth_token(
                        self.region
                    )
                    structlogger.debug(
                        "rasa.core.aws_msk_iam_credentials_provider.get_credentials",
                        event_info="Successfully generated AWS IAM token for "
                        "Kafka authentication.",
                    )
                    self.token = auth_token
                    self.expires_at = int(expiry_ms) / 1000  # Convert ms to seconds
                    return TemporaryCredentials(
                        auth_token=auth_token,
                        expiration=self.expires_at,
                    )
                except Exception as exc:
                    raise ConnectionException(
                        f"Failed to generate AWS IAM token "
                        f"for MSK authentication. Original exception: {exc}"
                    ) from exc
            else:
                structlogger.debug(
                    "rasa.core.aws_msk_iam_credentials_provider.get_credentials",
                    event_info="Using cached AWS IAM token for Kafka authentication.",
                )
                return TemporaryCredentials(
                    auth_token=self.token,
                    expiration=self.expires_at,
                )


def create_aws_iam_credentials_provider(
    provider_input: "IAMCredentialsProviderInput",
) -> Optional["IAMCredentialsProvider"]:
    """Factory function to create an AWS IAM credentials provider."""
    if provider_input.service_name == SupportedServiceType.TRACKER_STORE:
        return AWSRDSIAMCredentialsProvider(
            username=provider_input.username,
            host=provider_input.host,
            port=provider_input.port,
        )

    if provider_input.service_name == SupportedServiceType.EVENT_BROKER:
        return AWSMSKafkaIAMCredentialsProvider()

    return None
