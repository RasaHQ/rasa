import pytest
from _pytest.capture import CaptureFixture
from pytest import MonkeyPatch

from rasa.core.constants import (
    IAM_CLOUD_PROVIDER_ENV_VAR_NAME,
    RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME,
    SQL_SERVICE_NAME,
)
from rasa.core.iam_credentials_providers.aws_iam_credentials_providers import (
    AWSRDSIAMCredentialsProvider,
)
from rasa.core.iam_credentials_providers.credentials_provider_protocol import (
    IAMCredentialsProvider,
    IAMCredentialsProviderInput,
    SupportedServiceType,
    create_iam_credentials_provider,
)


@pytest.fixture
def provider_input(monkeypatch: MonkeyPatch) -> IAMCredentialsProviderInput:
    monkeypatch.setenv(RDS_SQL_DB_AWS_IAM_ENABLED_ENV_VAR_NAME, "true")
    return IAMCredentialsProviderInput(
        service_type=SupportedServiceType.TRACKER_STORE,
        service_name=SQL_SERVICE_NAME,
        username="test_user",
        host="localhost",
        port=5432,
    )


def test_create_iam_credentials_provider_with_aws(
    monkeypatch: MonkeyPatch,
    provider_input: IAMCredentialsProviderInput,
) -> None:
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "aws")
    provider = create_iam_credentials_provider(provider_input)
    assert isinstance(provider, IAMCredentialsProvider)
    assert isinstance(provider, AWSRDSIAMCredentialsProvider)
    assert provider.username == "test_user"
    assert provider.host == "localhost"
    assert provider.port == 5432


def test_create_iam_credentials_provider_with_unsupported_iam_provider(
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture,
    provider_input: IAMCredentialsProviderInput,
) -> None:
    monkeypatch.setenv(IAM_CLOUD_PROVIDER_ENV_VAR_NAME, "azure")

    provider = create_iam_credentials_provider(provider_input)
    assert provider is None

    captured = capsys.readouterr()
    assert "Unsupported IAM cloud provider: azure" in captured.out


def test_create_iam_credentials_provider_with_no_iam_provider(
    capsys: CaptureFixture,
    provider_input: IAMCredentialsProviderInput,
) -> None:
    provider = create_iam_credentials_provider(provider_input)
    assert provider is None
