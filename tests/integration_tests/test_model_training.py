import os.path
import tempfile
from pathlib import Path
from typing import Any, Text
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
from pytest import MonkeyPatch

import rasa.api
from rasa.core.agent import Agent
from rasa.env import BUCKET_NAME_ENV, REMOTE_STORAGE_PATH_ENV
from rasa.nlu.persistor import AWSPersistor, RemoteStorageType


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
    monkeypatch.setenv(BUCKET_NAME_ENV, bucket_name)
    monkeypatch.setenv("AWS_DEFAULT_REGION", region_name)

    # Moto uses these specific testing credentials
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    monkeypatch.setenv("TEST_SERVER_MODE", "true")
    yield


@pytest.fixture
def s3_connection(region_name: str, aws_environment_variables: None) -> Any:
    """Create a s3 connection to the Moto server."""
    with mock_aws():
        yield boto3.resource("s3", region_name=region_name)


@pytest.fixture
def setup_aws_persistor(
    s3_connection: Any,
    bucket_name: Text,
    region_name: Text,
    monkeypatch: MonkeyPatch,
) -> None:
    """Create an instance of the AWS persistor."""
    # We need to create the bucket in Moto's 'virtual' AWS account
    # prior to AWSPersistor instantiation
    s3_connection.create_bucket(
        Bucket=bucket_name,
    )

    aws_persistor = AWSPersistor(
        bucket_name,
        region_name=region_name,
    )
    monkeypatch.setattr(aws_persistor, "s3", s3_connection)
    monkeypatch.setattr(aws_persistor, "bucket", s3_connection.Bucket(bucket_name))

    _get_persistor = MagicMock()
    _get_persistor.return_value = aws_persistor

    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", _get_persistor)


@pytest.mark.parametrize(
    "remote_storage",
    [
        "",  # root of the bucket
        "some-remote-storage-path",
    ],
)
@pytest.mark.usefixtures("setup_aws_persistor")
def test_train_model_and_push_to_aws_remote_storage(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    empty_agent: Agent,
    domain_path: str,
    stories_path: str,
    stack_config_path: str,
    nlu_data_path: str,
    remote_storage: str,
) -> None:
    """Test to load model from AWS remote storage.

    Location of the model in the remote storage is provided
    through the environment variable REMOTE_STORAGE_PATH.
    """
    model_name = "dummy-model"
    empty_agent.remote_storage = RemoteStorageType.AWS
    monkeypatch.setenv(REMOTE_STORAGE_PATH_ENV, remote_storage)

    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")
    output = str(tmp_path / "models")

    rasa.api.train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
        fixed_model_name=model_name,
        remote_storage=RemoteStorageType.AWS,
        force_training=True,
    )
    empty_agent.load_model_from_remote_storage(
        f"{os.path.join(remote_storage, model_name)}.tar.gz"
    )
    assert empty_agent.processor.model_filename == f"{model_name}.tar.gz"


@pytest.mark.usefixtures("setup_aws_persistor")
def test_train_model_and_push_to_aws_remote_storage_without_remote_storage_env_var(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    empty_agent: Agent,
    domain_path: str,
    stories_path: str,
    stack_config_path: str,
    nlu_data_path: str,
) -> None:
    """Test to load model from AWS remote storage.

    Location of the model in the remote storage is not provided,
    it is consists of the output and model name in format output/model_name.tar.gz.
    """
    model_name = "dummy-model"
    empty_agent.remote_storage = RemoteStorageType.AWS
    monkeypatch.delenv(REMOTE_STORAGE_PATH_ENV, raising=False)

    (tmp_path / "training").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(tempfile, "tempdir", tmp_path / "training")
    output = str(tmp_path / "models")

    rasa.api.train(
        domain_path,
        stack_config_path,
        [stories_path, nlu_data_path],
        output=output,
        fixed_model_name=model_name,
        remote_storage=RemoteStorageType.AWS,
        force_training=True,
    )
    empty_agent.load_model_from_remote_storage(
        f"{os.path.join(output, model_name)}.tar.gz"
    )
    assert empty_agent.processor.model_filename == f"{model_name}.tar.gz"
