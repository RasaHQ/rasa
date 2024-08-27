import json
import os
import tarfile
from pathlib import Path
from typing import Any, Optional, Text, Union

import boto3
import pytest
from moto import mock_aws
from pytest import MonkeyPatch
from rasa.core.agent import Agent
from rasa.nlu.persistor import AWSPersistor

from rasa.e2e_test.e2e_test_runner import E2ETestRunner


@pytest.fixture
def mock_model(tmp_path: Path) -> Path:
    """Name of the model to use for testing."""
    model_file = tmp_path / "my-model.tar.gz"

    tarred_file = tmp_path / "dummy_file"
    tarred_file.touch()

    with tarfile.open(model_file, "w:gz") as tar:
        tar.add(tarred_file, arcname=tarred_file.name)

    return model_file


@pytest.fixture
def bucket_name() -> Text:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def region_name() -> Text:
    """Name of the region to use for testing."""
    return "us-east-1"


@mock_aws
def create_user_with_access_key_and_attached_policy(region_name: Text) -> Any:
    """Create a user and an access key for them."""
    client = boto3.client("iam", region_name=region_name)
    # deepcode ignore NoHardcodedCredentials/test: Test secret
    client.create_user(UserName="test_user")

    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllObjectActions",
                "Effect": "Allow",
                "Action": "s3:*Object",
                "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
            }
        ],
    }

    policy_arn = client.create_policy(
        PolicyName="test_policy", PolicyDocument=json.dumps(policy_document)
    )["Policy"]["Arn"]
    client.attach_user_policy(UserName="test_user", PolicyArn=policy_arn)

    return client.create_access_key(UserName="test_user")["AccessKey"]


@pytest.fixture
def aws_environment_variables(
    bucket_name: Text,
    region_name: Text,
) -> None:
    """Set AWS environment variables for testing."""
    os.environ["BUCKET_NAME"] = bucket_name
    os.environ["AWS_DEFAULT_REGION"] = region_name

    access_key = create_user_with_access_key_and_attached_policy(region_name)

    os.environ["AWS_ACCESS_KEY_ID"] = access_key["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = access_key["SecretAccessKey"]
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"

    os.environ["TEST_SERVER_MODE"] = "true"


def mock_load_model(
    self: Any, model_path: Union[Text, Path], fingerprint: Optional[Text] = None
) -> None:
    """Mock load model function."""

    class MockProcessor:
        def __init__(self, model_path: Any) -> None:
            self.model_path = Path(model_path)

    self.processor = MockProcessor(model_path)


@mock_aws
def test_e2e_test_runner_load_agent_from_remote_storage(
    mock_model: Path,
    bucket_name: Text,
    region_name: Text,
    aws_environment_variables: None,
    monkeypatch: MonkeyPatch,
) -> None:
    model_name = mock_model.name

    conn = boto3.resource("s3", region_name=region_name)
    # We need to create the bucket in Moto's 'virtual' AWS account
    # prior to AWSPersistor instantiation
    conn.create_bucket(Bucket=bucket_name)
    # upload model file to bucket
    with open(str(mock_model), "rb") as f:
        conn.meta.client.upload_fileobj(f, bucket_name, model_name)

    def mock_aws_persistor(name: Text) -> AWSPersistor:
        aws_persistor = AWSPersistor(bucket_name, region_name=region_name)
        monkeypatch.setattr(aws_persistor, "s3", conn)
        monkeypatch.setattr(aws_persistor, "bucket", conn.Bucket(bucket_name))
        return aws_persistor

    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", mock_aws_persistor)
    monkeypatch.setattr("rasa.core.agent.Agent.load_model", mock_load_model)

    test_runner = E2ETestRunner(model_path=model_name, remote_storage="aws")

    assert isinstance(test_runner.agent, Agent)
    assert test_runner.agent.remote_storage == "aws"

    assert test_runner.agent.processor is not None
    assert test_runner.agent.model_name is not None
