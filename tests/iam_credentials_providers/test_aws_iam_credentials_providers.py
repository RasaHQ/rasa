import time
from typing import Optional
from unittest.mock import MagicMock

import freezegun
import pytest
from moto import mock_aws
from moto.core import set_initial_no_auth_action_count
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSMSKafkaIAMCredentialsProvider,
    AWSRDSIAMCredentialsProvider,
    MSKAuthTokenProvider,
    create_aws_iam_credentials_provider,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
)
from rasa.shared.exceptions import ConnectionException


@pytest.fixture
def aws_rds_iam_provider_input() -> IAMCredentialsProviderInput:
    return IAMCredentialsProviderInput(
        service_name=SupportedServiceType.TRACKER_STORE,
        username="test_user",
        host="localhost",
        port=5432,
    )


def test_create_aws_iam_credentials_provider_for_tracker_store(
    aws_rds_iam_provider_input: IAMCredentialsProviderInput,
) -> None:
    iam_credentials_provider = create_aws_iam_credentials_provider(
        aws_rds_iam_provider_input
    )
    assert iam_credentials_provider is not None
    assert isinstance(iam_credentials_provider, IAMCredentialsProvider)
    assert isinstance(iam_credentials_provider, AWSRDSIAMCredentialsProvider)


@set_initial_no_auth_action_count(1)
@mock_aws
def test_aws_rds_iam_credentials_provider_get_credentials(
    aws_rds_iam_provider_input: IAMCredentialsProviderInput,
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_rds_iam_provider = create_aws_iam_credentials_provider(
        aws_rds_iam_provider_input
    )
    assert aws_rds_iam_provider is not None
    credentials = aws_rds_iam_provider.get_credentials()
    assert credentials.auth_token is not None
    assert "X-Amz-Credential" in credentials.auth_token

    captured = capsys.readouterr()
    assert "rasa.core.aws_rds_iam_credentials_provider.get_credentials" in captured.out
    assert (
        "rasa.core.aws_rds_iam_credentials_provider.generated_credentials"
        in captured.out
    )


@pytest.mark.parametrize(
    "username, host, port, expected_token_type",
    [
        (None, "localhost", 5432, str),
        ("test_user", None, 5432, str),
        ("test_user", "localhost", None, type(None)),
    ],
)
@set_initial_no_auth_action_count(1)
@mock_aws
def test_aws_rds_iam_credentials_provider_get_credentials_missing_input(
    username: Optional[str],
    host: Optional[str],
    port: Optional[int],
    expected_token_type: type,
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    provider_input = IAMCredentialsProviderInput(
        service_name=SupportedServiceType.TRACKER_STORE,
        username=username,
        host=host,
        port=port,
    )
    aws_rds_iam_provider = create_aws_iam_credentials_provider(provider_input)
    assert aws_rds_iam_provider is not None
    credentials = aws_rds_iam_provider.get_credentials()
    assert type(credentials.auth_token) == expected_token_type


def test_create_aws_iam_credentials_provider_for_kafka_broker() -> None:
    iam_credentials_provider = create_aws_iam_credentials_provider(
        IAMCredentialsProviderInput(
            service_name=SupportedServiceType.EVENT_BROKER,
        )
    )
    assert iam_credentials_provider is not None
    assert isinstance(iam_credentials_provider, IAMCredentialsProvider)
    assert isinstance(iam_credentials_provider, AWSMSKafkaIAMCredentialsProvider)


@set_initial_no_auth_action_count(1)
@mock_aws
def test_aws_msk_iam_credentials_provider_get_credentials(
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_kafka_iam_provider = create_aws_iam_credentials_provider(
        IAMCredentialsProviderInput(
            service_name=SupportedServiceType.EVENT_BROKER,
        )
    )
    assert aws_kafka_iam_provider is not None
    assert isinstance(aws_kafka_iam_provider, AWSMSKafkaIAMCredentialsProvider)
    assert aws_kafka_iam_provider.token is None

    credentials = aws_kafka_iam_provider.get_credentials()
    assert credentials.auth_token is not None
    assert isinstance(credentials.auth_token, str)
    assert credentials.expiration is not None
    assert credentials.expiration == aws_kafka_iam_provider.expires_at

    captured = capsys.readouterr()
    assert "rasa.core.aws_msk_iam_credentials_provider.get_credentials" in captured.out
    assert (
        "Successfully generated AWS IAM token for Kafka authentication." in captured.out
    )


# freeze time to ensure token refresh logic is tested
@freezegun.freeze_time("2023-01-01 00:00:00")
@set_initial_no_auth_action_count(1)
@mock_aws
def test_aws_msk_iam_credentials_provider_get_credentials_refresh_token(
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_kafka_iam_provider = create_aws_iam_credentials_provider(
        IAMCredentialsProviderInput(
            service_name=SupportedServiceType.EVENT_BROKER,
        )
    )
    assert aws_kafka_iam_provider is not None
    assert isinstance(aws_kafka_iam_provider, AWSMSKafkaIAMCredentialsProvider)
    monkeypatch.setattr(aws_kafka_iam_provider, "token", "existing_token")
    monkeypatch.setattr(
        aws_kafka_iam_provider, "expires_at", time.time() + 10
    )  # expires in 10 seconds

    credentials = aws_kafka_iam_provider.get_credentials()
    assert credentials.auth_token is not None
    assert credentials.auth_token != "existing_token"  # should have been refreshed
    assert credentials.expiration is not None
    assert credentials.expiration == aws_kafka_iam_provider.expires_at
    assert credentials.expiration != time.time() + 10  # should have been refreshed

    captured = capsys.readouterr()
    assert "rasa.core.aws_msk_iam_credentials_provider.get_credentials" in captured.out
    assert (
        "Successfully generated AWS IAM token for Kafka authentication." in captured.out
    )


# freeze time to ensure token refresh logic is tested
@freezegun.freeze_time("2023-01-01 00:00:00")
@set_initial_no_auth_action_count(1)
@mock_aws
def test_aws_msk_iam_credentials_provider_get_credentials_do_not_refresh_token(
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    aws_kafka_iam_provider = create_aws_iam_credentials_provider(
        IAMCredentialsProviderInput(
            service_name=SupportedServiceType.EVENT_BROKER,
        )
    )
    assert aws_kafka_iam_provider is not None
    assert isinstance(aws_kafka_iam_provider, AWSMSKafkaIAMCredentialsProvider)
    monkeypatch.setattr(aws_kafka_iam_provider, "token", "existing_token")
    monkeypatch.setattr(
        aws_kafka_iam_provider, "expires_at", time.time() + 120
    )  # expires in 120 seconds

    credentials = aws_kafka_iam_provider.get_credentials()
    assert credentials.auth_token is not None
    assert credentials.auth_token == "existing_token"  # should not have been refreshed
    assert credentials.expiration is not None
    assert credentials.expiration == aws_kafka_iam_provider.expires_at
    assert credentials.expiration == time.time() + 120  # should not have been refreshed
    captured = capsys.readouterr()
    assert "rasa.core.aws_msk_iam_credentials_provider.get_credentials" in captured.out
    assert "Using cached AWS IAM token for Kafka authentication." in captured.out


def test_aws_msk_iam_credentials_provider_get_credentials_raises_exception(
    capsys: CaptureFixture,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    mock_msk_auth_token_provider = MagicMock(
        spec=MSKAuthTokenProvider, side_effect=Exception("Test exception")
    )
    monkeypatch.setattr(
        MSKAuthTokenProvider, "generate_auth_token", mock_msk_auth_token_provider
    )

    aws_kafka_iam_provider = create_aws_iam_credentials_provider(
        IAMCredentialsProviderInput(
            service_name=SupportedServiceType.EVENT_BROKER,
        )
    )
    assert aws_kafka_iam_provider is not None
    assert isinstance(aws_kafka_iam_provider, AWSMSKafkaIAMCredentialsProvider)

    exception_msg = (
        "Failed to generate AWS IAM token for MSK authentication. "
        "Original exception: Test exception"
    )
    with pytest.raises(ConnectionException, match=exception_msg):
        aws_kafka_iam_provider.get_credentials()
