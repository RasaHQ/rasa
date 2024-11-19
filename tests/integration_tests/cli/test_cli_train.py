import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import boto3
import pytest
from _pytest.monkeypatch import MonkeyPatch
from moto import mock_aws

from rasa.cli import train
from rasa.cli.train import run_training
from rasa.nlu.persistor import AWSPersistor
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from tests.utilities import TarFileEntry, create_tar_archive_in_bytes


@pytest.fixture
def bucket_name() -> str:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def region_name() -> str:
    """Name of the region to use for testing."""
    return "us-east-1"


@pytest.fixture
def aws_endpoint_url() -> str:
    """URL of the moto testing server."""
    return "http://localhost:40000"


@pytest.fixture
def aws_environment_variables(
    bucket_name: str,
    region_name: str,
    aws_endpoint_url: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Set AWS environment variables for testing."""
    monkeypatch.setenv("BUCKET_NAME", bucket_name)
    monkeypatch.setenv("AWS_DEFAULT_REGION", region_name)

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")

    monkeypatch.setenv("TEST_SERVER_MODE", "true")
    yield


@pytest.fixture
def s3_connection(aws_environment_variables: None, region_name: str) -> Any:
    """Create an S3 connection."""
    with mock_aws():
        yield boto3.resource("s3", region_name=region_name)


@pytest.fixture
def aws_persistor(
    bucket_name: str,
    region_name: str,
    s3_connection: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Create an AWS persistor for testing.

    AWS persistor is using fake boto connection.
    """
    s3_connection.create_bucket(
        Bucket=bucket_name,
    )

    aws_persistor = AWSPersistor(
        os.environ.get("BUCKET_NAME"),
        region_name=os.environ.get("AWS_DEFAULT_REGION"),
    )
    monkeypatch.setattr(aws_persistor, "s3", s3_connection)
    monkeypatch.setattr(aws_persistor, "bucket", s3_connection.Bucket(bucket_name))

    _get_persistor_mock = MagicMock()
    _get_persistor_mock.return_value = aws_persistor
    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", _get_persistor_mock)


def create_in_memory_tar_archive_from_paths(paths: List[Path]) -> bytes:
    """Create a tar archive from a directory."""

    tar_entries: List[TarFileEntry] = []

    for path in paths:
        if path.is_file():
            tar_entries.append(TarFileEntry(str(path.name), path.read_bytes()))
        else:
            for root, _, files in os.walk(path):
                parent_directory = os.path.basename(root)
                for file in files:
                    tar_entries.append(
                        TarFileEntry(
                            os.path.join(parent_directory, file),
                            Path(os.path.join(root, file)).read_bytes(),
                        )
                    )

    return create_tar_archive_in_bytes(tar_entries)


@pytest.fixture
def default_config_file(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_CONFIG_PATH))


@pytest.fixture
def default_domain_file(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_DOMAIN_PATH))


@pytest.fixture
def default_data_directory(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_DATA_PATH))


@pytest.fixture
def in_memory_bot_config_tar_archive(
    project: str,
    default_config_file: Path,
    default_domain_file: Path,
    default_data_directory: Path,
) -> bytes:
    tar_archive_in_bytes = create_in_memory_tar_archive_from_paths(
        [default_config_file, default_domain_file, default_data_directory]
    )

    return tar_archive_in_bytes


@pytest.mark.usefixtures("aws_persistor")
def test_train_with_bot_config_from_remote_storage(
    bucket_name: str,
    s3_connection: Any,
    in_memory_bot_config_tar_archive: bytes,
    tmp_path: Path,
    tmp_path_factory: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test training with bot config from remote storage.

    Trained model will be saved on the cloud storage.
    """
    monkeypatch.chdir(
        tmp_path_factory.mktemp("train_with_bot_config_from_remote_storage")
    )
    bot_config_remote_path = "tests/fixtures/remote_storage_config.tar.gz"

    s3_connection.meta.client.put_object(
        Body=in_memory_bot_config_tar_archive,
        Bucket=bucket_name,
        Key=bot_config_remote_path,
        ContentType="application/tar+gzip",
    )

    # Create a parser for the `train` command
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    train.add_subparser(subparsers, parents=[])

    model_name = "test_model.tar.gz"
    args = parser.parse_args(
        [
            "train",
            "--remote-storage",
            "aws",
            "--fixed-model-name",
            f"{model_name}",
            "--out",
            f"{tmp_path}",
            "--remote-bot-config-path",
            bot_config_remote_path,
        ]
    )

    # Run the training with the bot config from the remote storage
    run_training(args)

    # Assert that the model is saved on the cloud storage
    response = s3_connection.meta.client.head_object(
        Bucket=bucket_name, Key=str(tmp_path / model_name)
    )
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200


@pytest.mark.usefixtures("aws_persistor")
def test_train_from_local_files_and_upload_trained_model(
    bucket_name: str,
    s3_connection: Any,
    tmp_path: Path,
    project: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test training with bot config from local storage.

    Trained model will be saved on the cloud storage.
    """
    # Switch to the project directory which contains the bot config
    monkeypatch.chdir(project)

    # Create a parser for the `train` command
    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    train.add_subparser(subparsers, parents=[])

    model_name = "test_model.tar.gz"
    args = parser.parse_args(
        [
            "train",
            "--remote-storage",
            "aws",
            "--fixed-model-name",
            f"{model_name}",
            "--out",
            f"{tmp_path}",
        ]
    )

    # Run the training with the bot config from the remote storage
    run_training(args)

    # Assert that the model is saved on the cloud storage
    response = s3_connection.meta.client.head_object(
        Bucket=bucket_name, Key=str(tmp_path / model_name)
    )
    assert response["ResponseMetadata"]["HTTPStatusCode"] == 200
