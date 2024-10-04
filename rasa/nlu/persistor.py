from __future__ import annotations

import abc
import os
import shutil
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Text, Tuple, Union

import structlog

import rasa.shared.utils.common
import rasa.utils.common
from rasa.constants import (
    HTTP_STATUS_FORBIDDEN,
    HTTP_STATUS_NOT_FOUND,
    MODEL_ARCHIVE_EXTENSION,
)
from rasa.env import (
    AWS_ENDPOINT_URL_ENV,
    AZURE_ACCOUNT_KEY_ENV,
    AZURE_ACCOUNT_NAME_ENV,
    AZURE_CONTAINER_ENV,
    BUCKET_NAME_ENV,
    REMOTE_STORAGE_PATH_ENV,
)
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import raise_warning

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient

structlogger = structlog.get_logger()


class RemoteStorageType(Enum):
    """Enum for the different remote storage types."""

    AWS = "aws"
    GCS = "gcs"
    AZURE = "azure"

    @classmethod
    def list(cls) -> List[str]:
        """Returns a list of all available storage types."""
        return [item.value for item in cls]


"""Storage can be a built-in one or a module path to a custom persistor."""
StorageType = Union[RemoteStorageType, str]


def parse_remote_storage(value: str) -> StorageType:
    try:
        return RemoteStorageType(value)
    except ValueError:
        # if the value is not a valid storage type,
        # but it is a string we assume it is a custom class
        # and return it as is

        supported_storages_help_text = (
            f"Supported storages are: {RemoteStorageType.list()} "
            "or path to a python class which implements `Persistor` interface."
        )

        if isinstance(value, str):
            if value == "":
                raise RasaException(
                    f"The value can't be an empty string."
                    f" {supported_storages_help_text}"
                )

            return value

        raise RasaException(
            f"Invalid storage type '{value}'. {supported_storages_help_text}"
        )


def get_persistor(storage: StorageType) -> Optional[Persistor]:
    """Returns an instance of the requested persistor.

    Currently, `aws`, `gcs`, `azure` and providing module paths are supported remote
    storages.
    """
    storage = storage.value if isinstance(storage, RemoteStorageType) else storage

    if storage == RemoteStorageType.AWS.value:
        return AWSPersistor(
            os.environ.get(BUCKET_NAME_ENV), os.environ.get(AWS_ENDPOINT_URL_ENV)
        )
    if storage == RemoteStorageType.GCS.value:
        return GCSPersistor(os.environ.get(BUCKET_NAME_ENV))

    if storage == RemoteStorageType.AZURE.value:
        return AzurePersistor(
            os.environ.get(AZURE_CONTAINER_ENV),
            os.environ.get(AZURE_ACCOUNT_NAME_ENV),
            os.environ.get(AZURE_ACCOUNT_KEY_ENV),
        )
    # If the persistor is not a built-in one, it is assumed to be a module path
    # to a persistor implementation supplied by the user.
    if storage:
        try:
            persistor = rasa.shared.utils.common.class_from_module_path(storage)
            return persistor()
        except ImportError:
            raise ImportError(
                f"Unknown model persistor {storage}. Please make sure to "
                f"either use an included model persistor ({RemoteStorageType.list()}) "
                f"or specify the module path to an external "
                "model persistor."
            )
    return None


class Persistor(abc.ABC):
    """Store models in cloud and fetch them when needed."""

    def persist(self, trained_model: str) -> None:
        """Uploads a trained model persisted in the `target_dir` to cloud storage."""
        file_key = self._create_file_key(trained_model)
        self._persist_tar(file_key, trained_model)

    def retrieve(self, model_name: Text, target_path: Text) -> Text:
        """Downloads a model that has been persisted to cloud storage.

        Downloaded model will be saved to the `target_path`.
        If `target_path` is a directory, the model will be saved to that directory.
        If `target_path` is a file, the model will be saved to that file.

        Args:
            model_name: The name of the model to retrieve.
            target_path: The path to which the model should be saved.
        """
        tar_name = model_name
        if not model_name.endswith(MODEL_ARCHIVE_EXTENSION):
            # ensure backward compatibility
            tar_name = self._tar_name(model_name)
        tar_name = self._create_file_key(tar_name)
        self._retrieve_tar(tar_name)
        self._copy(os.path.basename(tar_name), target_path)

        if os.path.isdir(target_path):
            return os.path.join(target_path, model_name)

        return target_path

    @abc.abstractmethod
    def _retrieve_tar(self, filename: Text) -> None:
        """Downloads a model previously persisted to cloud storage."""
        raise NotImplementedError

    @abc.abstractmethod
    def _persist_tar(self, filekey: Text, tarname: Text) -> None:
        """Uploads a model persisted in the `target_dir` to cloud storage."""
        raise NotImplementedError

    def _compress(self, model_directory: Text, model_name: Text) -> Tuple[Text, Text]:
        """Creates a compressed archive and returns key and tar."""
        import tempfile

        dirpath = tempfile.mkdtemp()
        base_name = self._tar_name(model_name, include_extension=False)
        tar_name = shutil.make_archive(
            os.path.join(dirpath, base_name),
            "gztar",
            root_dir=model_directory,
            base_dir=".",
        )
        file_key = os.path.basename(tar_name)
        return file_key, tar_name

    @staticmethod
    def _tar_name(model_name: Text, include_extension: bool = True) -> Text:
        ext = f".{MODEL_ARCHIVE_EXTENSION}" if include_extension else ""
        return f"{model_name}{ext}"

    @staticmethod
    def _copy(compressed_path: Text, target_path: Text) -> None:
        shutil.copy2(compressed_path, target_path)

    @staticmethod
    def _create_file_key(model_path: str) -> Text:
        """Appends remote storage folders when provided to upload or retrieve file"""
        bucket_object_path = os.environ.get(REMOTE_STORAGE_PATH_ENV)

        # To keep the backward compatibility, if REMOTE_STORAGE_PATH is not provided,
        # the model_name (which might be a complete path) will be returned as it is.
        if bucket_object_path is None:
            return str(model_path)
        else:
            raise_warning(
                f"{REMOTE_STORAGE_PATH_ENV} is deprecated and will be "
                "removed in future versions. "
                "Please use the -m path/to/model.tar.gz option to "
                "specify the model path when loading a model."
                "Or use --output and --fixed-model-name to specify the "
                "output directory and the model name when saving a "
                "trained model to remote storage.",
            )

        file_key = os.path.basename(model_path)
        file_key = os.path.join(bucket_object_path, file_key)
        return file_key


class AWSPersistor(Persistor):
    """Store models on S3.

    Fetches them when needed, instead of storing them on the local disk.
    """

    def __init__(
        self,
        bucket_name: Text,
        endpoint_url: Optional[Text] = None,
        region_name: Optional[Text] = None,
    ) -> None:
        import boto3

        super().__init__()
        self.s3 = boto3.resource(
            "s3", endpoint_url=endpoint_url, region_name=region_name
        )
        self._ensure_bucket_exists(bucket_name, region_name)
        self.bucket_name = bucket_name
        self.bucket = self.s3.Bucket(bucket_name)

    def _ensure_bucket_exists(
        self, bucket_name: Text, region_name: Optional[Text] = None
    ) -> None:
        import botocore

        # noinspection PyUnresolvedReferences
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket_name)
        except botocore.exceptions.ClientError as e:
            error_code = int(e.response["Error"]["Code"])
            if error_code == HTTP_STATUS_FORBIDDEN:
                log = (
                    f"Access to the specified bucket '{bucket_name}' is forbidden. "
                    "Please make sure you have the necessary "
                    "permission to access the bucket."
                )
                structlogger.error(
                    "aws_persistor.ensure_bucket_exists.bucket_access_forbidden",
                    bucket_name=bucket_name,
                    event_info=log,
                )
                raise RasaException(log)
            elif error_code == HTTP_STATUS_NOT_FOUND:
                log = (
                    f"The specified bucket '{bucket_name}' does not exist. "
                    "Please make sure to create the bucket first."
                )
                structlogger.error(
                    "aws_persistor.ensure_bucket_exists.bucket_not_found",
                    bucket_name=bucket_name,
                    event_info=log,
                )
                raise RasaException(log)

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to s3."""
        with open(tar_path, "rb") as f:
            self.s3.Object(self.bucket_name, file_key).put(Body=f)

    def _retrieve_tar(self, model_path: Text) -> None:
        """Downloads a model that has previously been persisted to s3."""
        tar_name = os.path.basename(model_path)
        with open(tar_name, "wb") as f:
            self.bucket.download_fileobj(model_path, f)


class GCSPersistor(Persistor):
    """Store models on Google Cloud Storage.

    Fetches them when needed, instead of storing them on the local disk.
    """

    def __init__(self, bucket_name: Text) -> None:
        """Initialise class with client and bucket."""
        # there are no type hints in this repo for now
        # https://github.com/googleapis/python-storage/issues/393
        from google.cloud import storage

        super().__init__()

        self.storage_client = storage.Client()
        self._ensure_bucket_exists(bucket_name)

        self.bucket_name = bucket_name
        self.bucket = self.storage_client.bucket(bucket_name)

    def _ensure_bucket_exists(self, bucket_name: Text) -> None:
        from google.cloud import exceptions

        try:
            self.storage_client.get_bucket(bucket_name)
        except exceptions.NotFound:
            log = (
                f"The specified bucket '{bucket_name}' does not exist. "
                "Please make sure to create the bucket first."
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.bucket_not_found",
                bucket_name=bucket_name,
                event_info=log,
            )
            raise RasaException(log)
        except exceptions.Forbidden:
            log = (
                f"Access to the specified bucket '{bucket_name}' is forbidden. "
                "Please make sure you have the necessary "
                "permission to access the bucket. "
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.bucket_access_forbidden",
                bucket_name=bucket_name,
                event_info=log,
            )
            raise RasaException(log)

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to GCS."""
        blob = self.bucket.blob(file_key)
        blob.upload_from_filename(tar_path)

    def _retrieve_tar(self, target_filename: Text) -> None:
        """Downloads a model that has previously been persisted to GCS."""
        blob = self.bucket.blob(target_filename)
        blob.download_to_filename(target_filename)


class AzurePersistor(Persistor):
    """Store models on Azure."""

    def __init__(
        self, azure_container: Text, azure_account_name: Text, azure_account_key: Text
    ) -> None:
        from azure.storage.blob import BlobServiceClient

        super().__init__()

        self.blob_service = BlobServiceClient(
            account_url=f"https://{azure_account_name}.blob.core.windows.net/",
            credential=azure_account_key,
        )
        self.container_name = azure_container
        self._ensure_container_exists()

    def _ensure_container_exists(self) -> None:
        if self._container_client().exists():
            pass
        else:
            log = (
                f"The specified container '{self.container_name}' does not exist."
                "Please make sure to create the container first."
            )
            structlogger.error(
                "azure_persistor.ensure_container_exists.container_not_found",
                container_name=self.container_name,
                event_info=log,
            )
            raise RasaException(log)

    def _container_client(self) -> "ContainerClient":
        return self.blob_service.get_container_client(self.container_name)

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to Azure."""
        with open(tar_path, "rb") as data:
            self._container_client().upload_blob(name=file_key, data=data)

    def _retrieve_tar(self, target_filename: Text) -> None:
        """Downloads a model that has previously been persisted to Azure."""
        blob_client = self._container_client().get_blob_client(target_filename)

        with open(target_filename, "wb") as blob:
            download_stream = blob_client.download_blob()
            blob.write(download_stream.readall())
