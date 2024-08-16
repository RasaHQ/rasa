import json
import os
from pathlib import Path
from typing import Any, Text
import tempfile
import boto3
import pytest
from moto import mock_iam, mock_s3
from pytest import MonkeyPatch

import rasa
from rasa.core.agent import Agent
from rasa.nlu.persistor import AWSPersistor
from rasa.shared.exceptions import RasaException


@pytest.fixture
def bucket_name() -> Text:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def region_name() -> Text:
    """Name of the region to use for testing."""
    return "us-west-2"


@pytest.fixture
def aws_endpoint_url() -> Text:
    """URL of the moto testing server."""
    return "http://localhost:5000"


@mock_iam
def create_user_with_access_key_and_attached_policy(region_name: Text) -> Any:
    """Create a user and an access key for them."""
    client = boto3.client("iam", region_name=region_name)
    # deepcode ignore NoHardcodedCredentials/test: Test credential
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
    aws_endpoint_url: Text,
) -> None:
    """Set AWS environment variables for testing."""
    os.environ["BUCKET_NAME"] = bucket_name
    os.environ["AWS_ENDPOINT_URL"] = aws_endpoint_url
    os.environ["AWS_DEFAULT_REGION"] = region_name

    access_key = create_user_with_access_key_and_attached_policy(region_name)

    os.environ["AWS_ACCESS_KEY_ID"] = access_key["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = access_key["SecretAccessKey"]
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"

    os.environ["TEST_SERVER_MODE"] = "true"


@mock_s3
def test_train_model_and_push_to_aws_remote_storage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    aws_environment_variables: Any,
    bucket_name: Text,
    region_name: Text,
    aws_endpoint_url: Text,
    empty_agent: Agent,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
) -> None:
    """Test to load model from AWS remote storage."""

    model_name = "dummy-model"

    conn = boto3.resource("s3", region_name=region_name)
    # We need to create the bucket in Moto's 'virtual' AWS account
    # prior to AWSPersistor instantiation
    conn.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": region_name},
    )

    def mock_aws_persistor(name: Text) -> AWSPersistor:
        aws_persistor = AWSPersistor(
            os.environ.get("BUCKET_NAME"),
            region_name=os.environ.get("AWS_DEFAULT_REGION"),
        )
        monkeypatch.setattr(aws_persistor, "s3", conn)
        monkeypatch.setattr(aws_persistor, "bucket", conn.Bucket(bucket_name))
        return aws_persistor

    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", mock_aws_persistor)

    empty_agent.remote_storage = "aws"

    try:
        (tmp_path / "training").mkdir()
        (tmp_path / "models").mkdir()

        monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")
        output = str(tmp_path / "models")

        rasa.train(
            domain_path,
            stack_config_path,
            [stories_path, nlu_data_path],
            output=output,
            fixed_model_name=model_name,
            remote_storage="aws",
            force_training=True,
        )
        empty_agent.load_model_from_remote_storage(f"{model_name}.tar.gz")
        assert empty_agent.processor.model_filename == f"{model_name}.tar.gz"

    except RasaException as exc:
        assert False, f"Test to load model from remote storage failed: {exc}"
