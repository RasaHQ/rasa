from typing import Text

import pytest
import boto3
from unittest.mock import patch, Mock

from moto import mock_aws

from rasa.nlu import persistor
from rasa.nlu.persistor import Persistor
from rasa.shared.exceptions import RasaException


class Object:
    pass


@pytest.fixture
def bucket_name() -> Text:
    """Name of the bucket to use for testing."""
    return "rasa-test"


@pytest.fixture
def model() -> Text:
    """Name of the model to use for testing."""
    return "/my/project/model.tar.gz"


@pytest.fixture
def destination() -> Text:
    """Target path to use for testing."""
    return "dst"


# noinspection PyPep8Naming
def test_retrieve_tar_archive_with_s3_namespace(
    bucket_name: Text, model: Text, destination: Text
):
    with mock_aws():
        conn = boto3.resource("s3")
        conn.create_bucket(Bucket=bucket_name)

        with patch.object(persistor.AWSPersistor, "_copy") as copy:
            with patch.object(persistor.AWSPersistor, "_retrieve_tar") as retrieve:
                persistor.AWSPersistor(bucket_name, region_name="foo").retrieve(
                    model, destination
                )
            copy.assert_called_once_with("model.tar.gz", destination)
            retrieve.assert_called_once_with(model)


# noinspection PyPep8Naming
def test_retrieve_tar_archive_with_s3_bucket_not_found(
    bucket_name: Text, model: Text, destination: Text
):
    with mock_aws():
        with patch.object(persistor.AWSPersistor, "_copy"):
            log = (
                f"The specified bucket '{bucket_name}' does not exist. "
                "Please make sure to create the bucket first."
            )
            with pytest.raises(RasaException, match=log):
                with patch.object(persistor.AWSPersistor, "_retrieve_tar"):
                    persistor.AWSPersistor(bucket_name, region_name="foo").retrieve(
                        model, destination
                    )


@patch("boto3.resource")
def test_retrieve_tar_archive_with_s3_bucket_forbidden(
    mock_resource: Mock, bucket_name: Text, monkeypatch
):
    import botocore

    aws_persistor = persistor.AWSPersistor(bucket_name)
    client = aws_persistor.s3.meta.client
    error = {"Error": {"Code": 403}}

    monkeypatch.setattr(
        client,
        "head_bucket",
        Mock(side_effect=botocore.exceptions.ClientError(error, "head_bucket")),
    )
    log = (
        f"Access to the specified bucket '{bucket_name}' is forbidden. "
        "Please make sure you have the necessary "
        "permission to access the bucket."
    )
    with pytest.raises(RasaException, match=log):
        aws_persistor._ensure_bucket_exists(bucket_name)
    mock_resource.assert_called_once()


# noinspection PyPep8Naming
def test_s3_private_retrieve_tar(bucket_name: Text, model: Text):
    with mock_aws():
        conn = boto3.resource("s3")
        conn.create_bucket(Bucket=bucket_name)
        # Ensure the S3 persistor writes to a filename `model.tar.gz`, whilst
        # passing the fully namespaced path to boto3
        awsPersistor = persistor.AWSPersistor(bucket_name, region_name="foo")

        with patch.object(awsPersistor.bucket, "download_fileobj") as download_fileobj:
            # noinspection PyProtectedMember
            awsPersistor._retrieve_tar(model)
        retrieveArgs = download_fileobj.call_args[0]
        assert retrieveArgs[0] == model
        assert retrieveArgs[1].name == "model.tar.gz"


class TestPersistor(Persistor):
    def _retrieve_tar(self, filename: Text) -> Text:
        pass

    def _persist_tar(self, filekey: Text, tarname: Text) -> None:
        pass


def test_get_external_persistor():
    p = persistor.get_persistor("tests.nlu.test_persistor.TestPersistor")
    assert isinstance(p, TestPersistor)


def test_raise_exception_in_get_external_persistor():
    with pytest.raises(ImportError):
        _ = persistor.get_persistor("unknown.persistor")


# noinspection PyPep8Naming
@pytest.mark.parametrize(
    "model, archive", [("model.tar.gz", "model.tar.gz"), ("model", "model.tar.gz")]
)
def test_retrieve_tar_archive(model: Text, archive: Text):
    with patch.object(TestPersistor, "_copy") as f:
        with patch.object(TestPersistor, "_retrieve_tar") as f:
            TestPersistor().retrieve(model, "dst")
        f.assert_called_once_with(archive)


@patch("google.cloud.storage.Client")
def test_retrieve_tar_archive_with_gcs_namespace(
    mock_client: Mock, bucket_name: Text, model: Text, destination: Text
):
    with patch.object(persistor.GCSPersistor, "_copy") as copy:
        with patch.object(persistor.GCSPersistor, "_retrieve_tar") as retrieve:
            persistor.GCSPersistor(bucket_name).retrieve(model, destination)
        copy.assert_called_once_with("model.tar.gz", destination)
        retrieve.assert_called_once_with(model)
    mock_client.assert_called_once()


@patch("google.cloud.storage.Client")
def test_retrieve_tar_archive_with_gcs_bucket_not_found(
    mock_client: Mock,
    bucket_name: Text,
):
    from google.cloud import exceptions

    gcs_persistor = persistor.GCSPersistor(bucket_name)
    gcs_persistor.storage_client.get_bucket = Mock(side_effect=exceptions.NotFound(""))

    log = (
        f"The specified bucket '{bucket_name}' does not exist. "
        "Please make sure to create the bucket first."
    )

    with pytest.raises(RasaException, match=log):
        gcs_persistor._ensure_bucket_exists(bucket_name)
    mock_client.assert_called_once()


@patch("google.cloud.storage.Client")
def test_retrieve_tar_archive_with_gcs_bucket_forbidden(
    mock_client: Mock,
    bucket_name: Text,
):
    from google.cloud import exceptions

    gcs_persistor = persistor.GCSPersistor(bucket_name)
    gcs_persistor.storage_client.get_bucket = Mock(side_effect=exceptions.Forbidden(""))

    log = (
        f"Access to the specified bucket '{bucket_name}' is forbidden. "
        "Please make sure you have the necessary "
        "permission to access the bucket. "
    )

    with pytest.raises(RasaException, match=log):
        gcs_persistor._ensure_bucket_exists(bucket_name)
    mock_client.assert_called_once()


@patch("azure.storage.blob.BlobServiceClient")
def test_retrieve_tar_archive_with_azure_namespace(
    mock_client: Mock, model: Text, destination: Text
):
    azure_persistor = persistor.AzurePersistor("foo", "bar", "3333")

    with patch.object(persistor.AzurePersistor, "_copy") as copy:
        with patch.object(azure_persistor, "_retrieve_tar") as retrieve:
            azure_persistor.retrieve(model, destination)
        copy.assert_called_once_with("model.tar.gz", destination)
        retrieve.assert_called_once_with(model)
    mock_client.assert_called_once()


@patch("azure.storage.blob.BlobServiceClient")
def test_retrieve_tar_archive_with_azure_bucket_not_found(
    mock_client: Mock, monkeypatch
):
    azure_persistor = persistor.AzurePersistor("foo", "bar", "foobar")
    azure_persistor.container_name = bucket_name
    container_client = azure_persistor._container_client()
    monkeypatch.setattr(container_client, "exists", Mock(return_value=False))

    log = (
        f"The specified container '{bucket_name}' does not exist."
        "Please make sure to create the container first."
    )
    with pytest.raises(RasaException, match=log):
        azure_persistor._ensure_container_exists()
    mock_client.assert_called_once()
