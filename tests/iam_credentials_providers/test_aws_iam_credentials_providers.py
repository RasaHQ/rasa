from typing import Optional

import pytest
from moto import mock_aws
from moto.core import set_initial_no_auth_action_count
from pytest import CaptureFixture, MonkeyPatch

from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSRDSIAMCredentialsProvider,
    create_aws_iam_credentials_provider,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
)


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
