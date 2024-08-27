import tempfile
from pathlib import Path
from typing import Any, Text
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
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
    return "us-east-1"


@pytest.fixture
def aws_environment_variables(
    bucket_name: Text,
    region_name: Text,
    monkeypatch: MonkeyPatch,
) -> None:
    """Set AWS environment variables for testing."""
    monkeypatch.setenv("BUCKET_NAME", bucket_name)
    monkeypatch.setenv("AWS_DEFAULT_REGION", region_name)

    # Moto uses these specific testing credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    monkeypatch.setenv("TEST_SERVER_MODE", "true")
    yield


@pytest.fixture
@mock_aws
def s3_connection(region_name: str, aws_environment_variables: None) -> Any:
    """Create a s3 connection to the Moto server."""
    return boto3.resource("s3", region_name=region_name)


@mock_aws
def test_train_model_and_push_to_aws_remote_storage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    empty_agent: Agent,
    domain_path: Text,
    stories_path: Text,
    stack_config_path: Text,
    nlu_data_path: Text,
    s3_connection: Any,
    bucket_name: Text,
    region_name: Text,
) -> None:
    """Test to load model from AWS remote storage."""
    # Create the bucket in Moto's 'virtual' AWS account
    s3_connection.create_bucket(Bucket=bucket_name)

    aws_persistor = AWSPersistor(bucket_name, region_name=region_name)
    monkeypatch.setattr(aws_persistor, "s3", s3_connection)
    monkeypatch.setattr(aws_persistor, "bucket", s3_connection.Bucket(bucket_name))

    _get_persistor = MagicMock()
    _get_persistor.return_value = aws_persistor
    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", _get_persistor)

    model_name = "dummy-model"
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
