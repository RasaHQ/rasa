import os
import threading
import time
from typing import Dict, Optional, Tuple
from urllib.parse import ParseResult, urlencode, urlunparse

import boto3
import redis
import structlog
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
from botocore.exceptions import BotoCoreError
from botocore.model import ServiceId
from botocore.session import get_session
from botocore.signers import RequestSigner
from cachetools import TTLCache, cached

from rasa.core.constants import (
    ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME,
    KAFKA_MSK_AWS_IAM_ENABLED_ENV_VAR_NAME,
    KAFKA_SERVICE_NAME,
    RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME,
    REDIS_SERVICE_NAME,
    SQL_SERVICE_NAME,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
    TemporaryCredentials,
)
from rasa.shared.exceptions import ConnectionException

structlogger = structlog.get_logger(__name__)

SERVICE_CONFIG: Dict[Tuple[SupportedServiceType, str], str] = {
    (
        SupportedServiceType.TRACKER_STORE,
        SQL_SERVICE_NAME,
    ): RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME,
    (
        SupportedServiceType.TRACKER_STORE,
        REDIS_SERVICE_NAME,
    ): ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME,
    (
        SupportedServiceType.EVENT_BROKER,
        KAFKA_SERVICE_NAME,
    ): KAFKA_MSK_AWS_IAM_ENABLED_ENV_VAR_NAME,
    (
        SupportedServiceType.LOCK_STORE,
        REDIS_SERVICE_NAME,
    ): ELASTICACHE_REDIS_AWS_IAM_ENABLED_ENV_VAR_NAME,
}


class AWSRDSIAMCredentialsProvider(IAMCredentialsProvider):
    """Generates temporary credentials for AWS RDS using IAM roles."""

    def __init__(self, username: str, host: str, port: int) -> None:
        """Initializes the provider."""
        self.username = username
        self.host = host
        self.port = port

    def get_temporary_credentials(self) -> TemporaryCredentials:
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

    def get_temporary_credentials(self) -> TemporaryCredentials:
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


class AWSElasticacheRedisIAMCredentialsProvider(redis.CredentialProvider):
    """Generates temporary credentials for AWS ElastiCache Redis using IAM roles."""

    def __init__(self, username: str, cluster_name: Optional[str] = None) -> None:
        """Initializes the provider."""
        self.username = username
        self.cluster_name = cluster_name
        self.region = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION"))
        self.session = get_session()
        self.request_signer = RequestSigner(
            ServiceId("elasticache"),
            self.region,
            "elasticache",
            "v4",
            self.session.get_credentials(),
            self.session.get_component("event_emitter"),
        )

    # Generated IAM tokens are valid for 15 minutes
    @cached(cache=TTLCache(maxsize=128, ttl=900))
    def get_credentials(self) -> Tuple[str, str]:
        """Generates temporary credentials for AWS ElastiCache Redis.

        Required method implementation by redis-py CredentialProvider parent class.
        Used internally by redis-py when connecting to Redis.
        """
        query_params = {"Action": "connect", "User": self.username}
        url = urlunparse(
            ParseResult(
                scheme="https",
                netloc=self.cluster_name,
                path="/",
                query=urlencode(query_params),
                params="",
                fragment="",
            )
        )
        signed_url = self.request_signer.generate_presigned_url(
            {"method": "GET", "url": url, "body": {}, "headers": {}, "context": {}},
            operation_name="connect",
            expires_in=900,
            region_name=self.region,
        )

        # RequestSigner only seems to work if the URL has a protocol, but
        # Elasticache only accepts the URL without a protocol
        # So strip it off the signed URL before returning
        return self.username, signed_url.removeprefix("https://")

    def get_temporary_credentials(self) -> TemporaryCredentials:
        """Generates temporary credentials for AWS ElastiCache Redis.

        Implemented to comply with the IAMCredentialsProvider rasa-pro interface.
        Calls the get_credentials method which is used internally by redis-py.
        """
        try:
            username, signed_url = self.get_credentials()
            structlogger.info(
                "rasa.core.aws_elasticache_redis_iam_credentials_provider.generated_credentials",
                event_info="Successfully generated temporary credentials for "
                "AWS ElastiCache Redis.",
            )
            return TemporaryCredentials(username=username, presigned_url=signed_url)
        except Exception as exc:
            structlogger.error(
                "rasa.core.aws_elasticache_redis_iam_credentials_provider.error_generating_credentials",
                event_info="Failed to generate temporary credentials for "
                "AWS ElastiCache Redis.",
                error=str(exc),
            )
            return TemporaryCredentials()


def is_iam_enabled(provider_input: "IAMCredentialsProviderInput") -> bool:
    """Checks if IAM authentication is enabled for the given service."""
    service_type = provider_input.service_type
    service_name = provider_input.service_name
    iam_enabled_env_var_name = SERVICE_CONFIG.get((service_type, service_name))

    if not iam_enabled_env_var_name:
        structlogger.warning(
            "rasa.core.aws_iam_credentials_providers.is_iam_enabled.unsupported_service",
            event_info=f"IAM authentication check requested for unsupported service: "
            f"{service_name}",
        )
        return False

    return os.getenv(iam_enabled_env_var_name, "false").lower() == "true"


def create_aws_iam_credentials_provider(
    provider_input: "IAMCredentialsProviderInput",
) -> Optional["IAMCredentialsProvider"]:
    """Factory function to create an AWS IAM credentials provider."""
    iam_enabled = is_iam_enabled(provider_input)
    if not iam_enabled:
        structlogger.debug(
            "rasa.core.aws_iam_credentials_providers.create_provider.iam_not_enabled",
            event_info=f"IAM authentication not enabled for service: "
            f"{provider_input.service_type}",
        )
        return None

    if (
        provider_input.service_type == SupportedServiceType.TRACKER_STORE
        and provider_input.service_name == SQL_SERVICE_NAME
    ):
        return AWSRDSIAMCredentialsProvider(
            username=provider_input.username,
            host=provider_input.host,
            port=provider_input.port,
        )

    if (
        provider_input.service_type == SupportedServiceType.TRACKER_STORE
        and provider_input.service_name == REDIS_SERVICE_NAME
    ):
        return AWSElasticacheRedisIAMCredentialsProvider(
            username=provider_input.username,
            cluster_name=provider_input.cluster_name,
        )

    if provider_input.service_type == SupportedServiceType.EVENT_BROKER:
        return AWSMSKafkaIAMCredentialsProvider()

    if provider_input.service_type == SupportedServiceType.LOCK_STORE:
        return AWSElasticacheRedisIAMCredentialsProvider(
            username=provider_input.username,
            cluster_name=provider_input.cluster_name,
        )

    return None
