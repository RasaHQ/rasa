import os
from typing import Any, Dict, Text
from unittest.mock import MagicMock, Mock, patch

import boto3
import pytest
from _pytest.monkeypatch import MonkeyPatch
from moto import mock_aws

from rasa.env import (
    AZURE_ACCOUNT_KEY_ENV,
    AZURE_ACCOUNT_NAME_ENV,
    AZURE_CONTAINER_ENV,
    BUCKET_NAME_ENV,
    REMOTE_STORAGE_PATH_ENV,
)
from rasa.nlu import persistor
from rasa.nlu.persistor import (
    AWSPersistor,
    AzurePersistor,
    GCSPersistor,
    Persistor,
    RemoteStorageType,
    get_persistor,
)
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


@pytest.fixture
def mock_s3_connection() -> Any:
    with mock_aws():
        conn = boto3.resource("s3")
        yield conn


# noinspection PyPep8Naming
def test_retrieve_tar_archive_with_s3_namespace(
    bucket_name: Text, model: Text, destination: Text, mock_s3_connection: Any
):
    mock_s3_connection.create_bucket(Bucket=bucket_name)

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
def test_s3_private_retrieve_tar(
    bucket_name: Text, model: Text, mock_s3_connection: Any
):
    mock_s3_connection.create_bucket(Bucket=bucket_name)
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


@pytest.mark.parametrize(
    "model_name, expected_file_key, remote_storage_path",
    [
        ("model1.pkl", "model1.pkl", ""),
        ("model1.pkl", "test_model/model1.pkl", "test_model/"),
    ],
)
def test_create_file_key_with_remote_path(
    model_name: Text, expected_file_key: Text, remote_storage_path: Text
):
    # Simulate the environment where BUCKET_OBJECT_PATH is set
    with patch.dict(os.environ, {"REMOTE_STORAGE_PATH": remote_storage_path}):
        result = TestPersistor()._create_file_key(model_name)
        assert result == expected_file_key


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


@pytest.mark.parametrize(
    "model_path, envs, expected_file_key",
    [
        ("model1.pkl", {}, "model1.pkl"),
        ("path/to/file/model1.pkl", {}, "path/to/file/model1.pkl"),
        (
            "model1.pkl",
            {REMOTE_STORAGE_PATH_ENV: "test_model"},
            "test_model/model1.pkl",
        ),
        (
            "path/to/file/model1.pkl",
            {
                REMOTE_STORAGE_PATH_ENV: "test_model",
            },
            "test_model/model1.pkl",
        ),
    ],
)
def test_create_file_key(
    model_path: Text,
    envs: Dict[str, str],
    expected_file_key: Text,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test file key creation.

    File key is the file path in the storage bucket.
    """

    monkeypatch.delenv(REMOTE_STORAGE_PATH_ENV, raising=False)
    for key, value in envs.items():
        monkeypatch.setenv(key, value)

    file_key = Persistor._create_file_key(model_path)
    assert file_key == expected_file_key


def test_create_file_key_remote_storage_path_deprecation_logging(
    caplog: Any,
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(REMOTE_STORAGE_PATH_ENV, "test_model")
    mock_raise_warning = MagicMock()
    monkeypatch.setattr("rasa.nlu.persistor.raise_warning", mock_raise_warning)
    Persistor._create_file_key("model1.pkl")
    warning_text = (
        f"{REMOTE_STORAGE_PATH_ENV} is deprecated and will be "
        "removed in future versions. "
        "Please use the -m path/to/model.tar.gz option to "
        "specify the model path when loading a model."
        "Or use --output and --fixed-model-name to specify the "
        "output directory and the model name when saving a "
        "trained model to remote storage."
    )
    mock_raise_warning.assert_called_once_with(warning_text)


def test_get_persistor_for_aws_remote_storage(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "rasa.nlu.persistor.AWSPersistor._ensure_bucket_exists",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setenv(BUCKET_NAME_ENV, "test_bucket")

    persistor_obj = get_persistor(RemoteStorageType("aws"))
    assert isinstance(persistor_obj, AWSPersistor)


def test_get_persistor_for_gcs_remote_storage(
    monkeypatch: MonkeyPatch,
) -> None:
    mock_gcs_client = MagicMock()
    mock_gcs_client.bucket.return_value = MagicMock()
    monkeypatch.setattr("google.cloud.storage.Client", mock_gcs_client)

    monkeypatch.setattr(
        "rasa.nlu.persistor.GCSPersistor._ensure_bucket_exists",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setenv(BUCKET_NAME_ENV, "test_bucket")

    persistor_obj = get_persistor(RemoteStorageType("gcs"))
    assert isinstance(persistor_obj, GCSPersistor)


def test_get_persistor_with_custom_persistor() -> None:
    persistor_obj = get_persistor("tests.nlu.test_persistor.TestPersistor")
    assert isinstance(persistor_obj, TestPersistor)


def test_get_persistor_for_azure_remote_storage(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv(AZURE_CONTAINER_ENV, "test_container")
    monkeypatch.setenv(AZURE_ACCOUNT_NAME_ENV, "test_account")
    monkeypatch.setenv(AZURE_ACCOUNT_KEY_ENV, "test_key")
    monkeypatch.setattr(
        AzurePersistor, "_ensure_container_exists", lambda *args, **kwargs: None
    )

    persistor_obj = get_persistor(RemoteStorageType("azure"))
    assert isinstance(persistor_obj, AzurePersistor)


def test_get_persistor_with_unknown_storage_type() -> None:
    with pytest.raises(ImportError, match="Unknown model persistor"):
        get_persistor("unknown")
