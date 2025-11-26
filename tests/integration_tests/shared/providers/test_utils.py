import contextlib
import json

import boto3
import pytest
import structlog.testing
from _pytest.monkeypatch import MonkeyPatch
from moto import mock_aws
from moto.core import set_initial_no_auth_action_count

from rasa.shared.constants import (
    AWS_ACCESS_KEY_ID_ENV_VAR,
    AWS_REGION_NAME_CONFIG_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_VAR,
    AWS_SESSION_TOKEN_ENV_VAR,
    LLM_API_HEALTH_CHECK_ENV_VAR,
)
from rasa.shared.providers._utils import validate_aws_setup_for_litellm_clients
from tests.utilities import filter_logs


@pytest.fixture
def region_name() -> str:
    """Name of the region to use for testing."""
    return "us-east-1"


@pytest.fixture
def aws_bedrock_policy() -> str:
    """AWS policy for testing."""
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": "*"},
            {
                "Effect": "Allow",
                "Action": ["bedrock:GetModelInvocationLoggingConfiguration"],
                "Resource": "*",
            },
        ],
    }
    return json.dumps(policy_document)


@pytest.fixture
def aws_sagemaker_policy() -> str:
    """AWS policy for testing."""
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": ["sagemaker:ListModels"], "Resource": "*"}
        ],
    }
    return json.dumps(policy_document)


@pytest.mark.parametrize("provider", ["bedrock", "sagemaker"])
@set_initial_no_auth_action_count(5)
@mock_aws
def test_validate_aws_setup_for_litellm_clients(
    provider: str,
    region_name: str,
    aws_bedrock_policy: str,
    aws_sagemaker_policy: str,
    monkeypatch: MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv(LLM_API_HEALTH_CHECK_ENV_VAR, "true")
    iam_client = boto3.client("iam", region_name=region_name)
    sts_client = boto3.client("sts", region_name=region_name)

    if provider == "bedrock":
        aws_policy = aws_bedrock_policy
    else:
        aws_policy = aws_sagemaker_policy

    account_id = "123456789012"
    trust_policy_document = {
        "Version": "2012-10-17",
        "Statement": {
            "Effect": "Allow",
            "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
            "Action": "sts:AssumeRole",
        },
    }
    role_name = "test_role"
    role_arn = iam_client.create_role(
        RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_policy_document)
    )["Role"]["Arn"]
    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="test-policy",
        PolicyDocument=aws_policy,
    )
    credentials = sts_client.assume_role(
        RoleArn=role_arn, RoleSessionName="test-session"
    )["Credentials"]
    monkeypatch.setenv(AWS_ACCESS_KEY_ID_ENV_VAR, credentials["AccessKeyId"])
    monkeypatch.setenv(AWS_SECRET_ACCESS_KEY_ENV_VAR, credentials["SecretAccessKey"])
    monkeypatch.setenv(AWS_SESSION_TOKEN_ENV_VAR, credentials["SessionToken"])

    litellm_model_name = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    call_kwargs = {
        "model_id": f"arn:aws:bedrock:{region_name}:{account_id}:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",  # noqa: E501
        AWS_REGION_NAME_CONFIG_KEY: region_name,
    }

    # Act & Assert
    with contextlib.nullcontext():
        with structlog.testing.capture_logs() as capture_logs:
            validate_aws_setup_for_litellm_clients(
                litellm_model_name=litellm_model_name,
                litellm_call_kwargs=call_kwargs,
                source_log="test_function",
                provider=provider,
            )
            debug_logs = filter_logs(
                capture_logs,
                event="test_function.validating_aws_credentials_for_litellm_clients_via_aws_client",
                log_level="debug",
            )
            assert len(debug_logs) == 1

            error_logs = filter_logs(
                capture_logs,
                event="test_function.validate_aws_credentials_for_litellm_clients_via_aws_client.failed",
                log_level="error",
            )
            assert len(error_logs) == 0
