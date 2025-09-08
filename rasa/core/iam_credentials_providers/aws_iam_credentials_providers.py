from typing import Optional

import boto3
import structlog
from botocore.exceptions import BotoCoreError

from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
    TemporaryCredentials,
)

structlogger = structlog.get_logger(__name__)


class AWSRDSIAMCredentialsProvider(IAMCredentialsProvider):
    """Generates temporary credentials for AWS RDS using IAM roles."""

    def __init__(self, username: str, host: str, port: int):
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

    return None
