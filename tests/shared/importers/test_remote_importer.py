import json
import os
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws
from pytest import MonkeyPatch

from rasa.nlu.persistor import AWSPersistor
from rasa.shared.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DOMAIN_PATH,
)
from rasa.shared.core.constants import (
    DEFAULT_ACTION_NAMES,
    DEFAULT_INTENTS,
    DEFAULT_SLOT_NAMES,
    REQUESTED_SLOT,
)
from rasa.shared.core.slots import AnySlot
from rasa.shared.importers.remote_importer import RemoteTrainingDataImporter
from tests.utilities import TarFileEntry, create_tar_archive_in_bytes


@pytest.fixture
def bucket_name() -> str:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def region_name() -> str:
    """Name of the region to use for testing."""
    return "us-east-1"


def create_user_with_access_key_and_attached_policy(region_name: str) -> Any:
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
    bucket_name: str,
    region_name: str,
    monkeypatch: MonkeyPatch,
) -> None:
    """Set AWS environment variables for testing."""
    monkeypatch.setenv("BUCKET_NAME", bucket_name)
    monkeypatch.setenv("AWS_DEFAULT_REGION", region_name)

    # access_key = create_user_with_access_key_and_attached_policy(region_name)

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


def create_in_memory_tar_archive_from_paths(paths: List[Path]) -> bytes:
    """Create a tar archive from a directory."""

    file_paths: List[Path] = []

    for path in paths:
        if path.is_file():
            file_paths.append(path)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = Path(os.path.join(root, file))
                    file_paths.append(file_path)

    tar_entries = [
        TarFileEntry(str(file_path.name), file_path.read_bytes())
        for file_path in file_paths
    ]

    return create_tar_archive_in_bytes(tar_entries)


@pytest.fixture
def default_config_file(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_CONFIG_PATH))


INTENT_COUNT_FROM_DEFAULT_DOMAIN = 7
RESPONSE_COUNT_FROM_DEFAULT_DOMAIN = 6
ACTION_COUNT_FROM_DEFAULT_DOMAIN = 6
STORY_STEP_COUNT_FROM_DEFAULT_STORIES = 5
NLU_INTENT_EXAMPLE_COUNT = 68
NLU_DATA_INTENT_COUNT = 7


@pytest.fixture
def default_domain_file(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_DOMAIN_PATH))


@pytest.fixture
def default_data_directory(project: str) -> Path:
    return Path(os.path.join(project, DEFAULT_DATA_PATH))


@pytest.fixture
def in_memory_bot_config_tar_archive(
    project: str,
    default_domain_file: Path,
    default_data_directory: Path,
) -> bytes:
    tar_archive_in_bytes = create_in_memory_tar_archive_from_paths(
        [default_domain_file, default_data_directory]
    )

    return tar_archive_in_bytes


@mock_aws
def test_remote_file_importer(
    monkeypatch: MonkeyPatch,
    bucket_name: str,
    region_name: str,
    project: str,
    s3_connection: Any,
    tmp_path: Path,
    default_config_file: Path,
    in_memory_bot_config_tar_archive: bytes,
):
    """Test the RemoteTrainingDataImporter.

    A default project files are packed into a
    tar archive and uploaded to an S3 bucket.

    The RemoteTrainingDataImporter is then used to
    download the tar archive, extract the files and
    load domain, stories, nlu data and conversation tests
    from the extracted files.
    """
    s3_connection.create_bucket(Bucket=bucket_name)

    aws_persistor = AWSPersistor(bucket_name, region_name=region_name)
    monkeypatch.setattr(aws_persistor, "s3", s3_connection)
    monkeypatch.setattr(aws_persistor, "bucket", s3_connection.Bucket(bucket_name))

    _get_persistor_mock = MagicMock()
    _get_persistor_mock.return_value = aws_persistor
    monkeypatch.setattr("rasa.nlu.persistor.get_persistor", _get_persistor_mock)

    # Given the tar archive with bot config files
    # is uploaded to the S3 bucket
    s3_connection.meta.client.put_object(
        Body=in_memory_bot_config_tar_archive,
        Bucket=bucket_name,
        Key="training_data.tar.gz",
        ContentType="application/tar+gzip",
    )

    # When the RemoteTrainingDataImporter is used to
    # download the tar archive, extract the files and
    # load domain, stories, nlu data and conversation tests
    importer = RemoteTrainingDataImporter(
        config_file=str(default_config_file),
        remote_storage="aws",
        training_data_path=str(tmp_path),
    )

    # Then the domain, stories, nlu data and conversation tests
    # are loaded from the extracted files
    domain = importer.get_domain()
    assert len(domain.intents) == INTENT_COUNT_FROM_DEFAULT_DOMAIN + len(
        DEFAULT_INTENTS
    )
    default_slots = [
        AnySlot(slot_name, mappings=[{}])
        for slot_name in DEFAULT_SLOT_NAMES
        if slot_name != REQUESTED_SLOT
    ]
    assert sorted(domain.slots, key=lambda s: s.name) == sorted(
        default_slots, key=lambda s: s.name
    )

    assert domain.entities == []
    assert len(domain.action_names_or_texts) == ACTION_COUNT_FROM_DEFAULT_DOMAIN + len(
        DEFAULT_ACTION_NAMES
    )
    assert len(domain.responses) == RESPONSE_COUNT_FROM_DEFAULT_DOMAIN

    stories = importer.get_stories()
    assert len(stories.story_steps) == STORY_STEP_COUNT_FROM_DEFAULT_STORIES

    test_stories = importer.get_conversation_tests()
    assert len(test_stories.story_steps) == 0

    nlu_data = importer.get_nlu_data("en")
    assert len(nlu_data.intents) == NLU_DATA_INTENT_COUNT
    assert len(nlu_data.intent_examples) == NLU_INTENT_EXAMPLE_COUNT
