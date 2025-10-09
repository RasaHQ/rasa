from __future__ import annotations

import abc
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Text, Tuple, Union

import structlog

import rasa.shared.utils.common
import rasa.utils.common
from rasa.constants import (
    DEFAULT_BUCKET_NAME,
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
from rasa.exceptions import ModelNotFound
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import raise_warning

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient
    from botocore.exceptions import ClientError

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
            os.environ.get(BUCKET_NAME_ENV, DEFAULT_BUCKET_NAME),
            os.environ.get(AWS_ENDPOINT_URL_ENV),
        )
    if storage == RemoteStorageType.GCS.value:
        return GCSPersistor(os.environ.get(BUCKET_NAME_ENV, DEFAULT_BUCKET_NAME))

    if storage == RemoteStorageType.AZURE.value:
        return AzurePersistor(
            os.environ.get(AZURE_CONTAINER_ENV, DEFAULT_BUCKET_NAME),
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

    def persist(self, trained_model: str, remote_root_only: bool = False) -> None:
        """Uploads a trained model persisted in the `target_dir` to cloud storage."""
        absolute_file_key = self._create_file_key(trained_model)
        file_key = (
            Path(absolute_file_key).name if remote_root_only else absolute_file_key
        )
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
        self._retrieve_tar(tar_name, target_path)

        if os.path.isdir(target_path):
            return os.path.join(target_path, model_name)

        return target_path

    def size_of_persisted_model(
        self, model_name: Text, target_path: Optional[str] = None
    ) -> int:
        """Returns the size of the model that has been persisted to cloud storage.

        Args:
            model_name: The name of the model to retrieve.
            target_path: The path to which the model should be saved.
        """
        tar_name = model_name
        if not model_name.endswith(MODEL_ARCHIVE_EXTENSION):
            # ensure backward compatibility
            tar_name = self._tar_name(model_name)
        tar_name = self._create_file_key(tar_name)
        return self._retrieve_tar_size(tar_name, target_path)

    def _retrieve_tar_size(
        self, filename: Text, target_path: Optional[str] = None
    ) -> int:
        """Returns the size of the model that has been persisted to cloud storage."""
        structlogger.warning(
            "persistor.retrieve_tar_size.not_implemented",
            filename=filename,
            event_info=(
                "This method should be implemented in the persistor. "
                "The default implementation will download the model "
                "to calculate the size. Most persistors should override "
                "this method to avoid downloading the model and get the "
                "size directly from the cloud storage."
            ),
        )
        self._retrieve_tar(filename, target_path)
        return os.path.getsize(os.path.basename(filename))

    @abc.abstractmethod
    def _retrieve_tar(self, filename: str, target_path: Optional[str] = None) -> None:
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
            base_dir="../nlu",
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
        """Appends remote storage folders when provided to upload or retrieve file."""
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
                "specify the model path when loading a model.",
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
        from botocore import exceptions

        # noinspection PyUnresolvedReferences
        try:
            self.s3.meta.client.head_bucket(Bucket=bucket_name)
        except exceptions.ClientError as exc:
            if self._error_code(exc) == HTTP_STATUS_FORBIDDEN:
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
            elif self._error_code(exc) == HTTP_STATUS_NOT_FOUND:
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

    @staticmethod
    def _error_code(e: "ClientError") -> int:
        return int(e.response["Error"]["Code"])

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to s3."""
        with open(tar_path, "rb") as f:
            self.s3.Object(self.bucket_name, file_key).put(Body=f)

    def _retrieve_tar_size(
        self, model_path: Text, target_path: Optional[str] = None
    ) -> int:
        """Returns the size of the model that has been persisted to s3."""
        try:
            obj = self.s3.Object(self.bucket_name, model_path)
            return obj.content_length
        except Exception:
            raise ModelNotFound("Model not found")

    def _retrieve_tar(
        self, target_filename: str, target_path: Optional[str] = None
    ) -> None:
        """Downloads a model that has previously been persisted to s3."""
        from botocore import exceptions

        log = (
            f"Model '{target_filename}' not found in the specified bucket "
            f"'{self.bucket_name}'. Please make sure the model exists "
            f"in the bucket."
        )

        tar_name = (
            os.path.join(target_path, os.path.basename(target_filename))
            if target_path
            else os.path.basename(target_filename)
        )

        try:
            with open(tar_name, "wb") as f:
                self.bucket.download_fileobj(target_filename, f)

            structlogger.debug(
                "aws_persistor.retrieve_tar.object_found", object_key=target_filename
            )
        except exceptions.ClientError as exc:
            if self._error_code(exc) == HTTP_STATUS_NOT_FOUND:
                structlogger.error(
                    "aws_persistor.retrieve_tar.model_not_found",
                    bucket_name=self.bucket_name,
                    target_filename=target_filename,
                    event_info=log,
                )
                raise ModelNotFound("Model not found") from exc
        except exceptions.BotoCoreError as exc:
            structlogger.error(
                "aws_persistor.retrieve_tar.model_download_error",
                bucket_name=self.bucket_name,
                target_filename=target_filename,
                event_info=log,
            )
            raise ModelNotFound("Model not found") from exc


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
        from google.auth import exceptions as auth_exceptions
        from google.cloud import exceptions

        try:
            self.storage_client.get_bucket(bucket_name)
        except auth_exceptions.GoogleAuthError as exc:
            log = (
                f"An error occurred while authenticating with Google Cloud "
                f"Storage. Please make sure you have the necessary credentials "
                f"to access the bucket '{bucket_name}'."
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.authentication_error",
                bucket_name=bucket_name,
                event_info=log,
            )
            raise RasaException(log) from exc
        except exceptions.NotFound as exc:
            log = (
                f"The specified Google Cloud Storage bucket '{bucket_name}' "
                f"does not exist. Please make sure to create the bucket first or "
                f"provide an alternative valid bucket name."
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.bucket_not_found",
                bucket_name=bucket_name,
                event_info=log,
            )
            raise RasaException(log) from exc
        except exceptions.Forbidden as exc:
            log = (
                f"Access to the specified Google Cloud storage bucket '{bucket_name}' "
                f"is forbidden. Please make sure you have the necessary "
                f"permissions to access the bucket. "
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.bucket_access_forbidden",
                bucket_name=bucket_name,
                event_info=log,
            )
            raise RasaException(log) from exc
        except ValueError as exc:
            # bucket_name is None
            log = (
                "The specified Google Cloud Storage bucket name is None. Please "
                "make sure to provide a valid bucket name."
            )
            structlogger.error(
                "gcp_persistor.ensure_bucket_exists.bucket_name_none",
                event_info=log,
            )
            raise RasaException(log) from exc

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to GCS."""
        blob = self.bucket.blob(file_key)
        blob.upload_from_filename(tar_path)

    def _retrieve_tar_size(
        self, target_filename: Text, target_path: Optional[str] = None
    ) -> int:
        """Returns the size of the model that has been persisted to GCS."""
        try:
            blob = self.bucket.blob(target_filename)
            return blob.size
        except Exception:
            raise ModelNotFound("Model not found")

    def _retrieve_tar(
        self, target_filename: str, target_path: Optional[str] = None
    ) -> None:
        """Downloads a model that has previously been persisted to GCS."""
        from google.api_core import exceptions

        blob = self.bucket.blob(target_filename)

        destination = (
            os.path.join(target_path, os.path.basename(target_filename))
            if target_path
            else target_filename
        )

        try:
            blob.download_to_filename(destination)

            structlogger.debug(
                "gcs_persistor.retrieve_tar.object_found", object_key=target_filename
            )
        except exceptions.NotFound as exc:
            log = (
                f"Model '{target_filename}' not found in the specified bucket "
                f"'{self.bucket_name}'. Please make sure the model exists "
                f"in the bucket."
            )
            structlogger.error(
                "gcp_persistor.retrieve_tar.model_not_found",
                bucket_name=self.bucket_name,
                target_filename=target_filename,
                event_info=log,
            )
            raise ModelNotFound("Model not found") from exc


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
                "Please make sure to create the bucket first or "
                f"provide an alternative valid bucket name."
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

    def _retrieve_tar_size(
        self, target_filename: Text, target_path: Optional[str] = None
    ) -> int:
        """Returns the size of the model that has been persisted to Azure."""
        try:
            blob_client = self._container_client().get_blob_client(target_filename)
            properties = blob_client.get_blob_properties()
            return properties.size
        except Exception:
            raise ModelNotFound("Model not found")

    def _retrieve_tar(
        self, target_filename: Text, target_path: Optional[str] = None
    ) -> None:
        """Downloads a model that has previously been persisted to Azure."""
        from azure.core.exceptions import AzureError

        destination = (
            os.path.join(target_path, os.path.basename(target_filename))
            if target_path
            else target_filename
        )

        try:
            with open(destination, "wb") as model_file:
                blob_client = self._container_client().get_blob_client(target_filename)
                download_stream = blob_client.download_blob()
                model_file.write(download_stream.readall())
            structlogger.debug(
                "azure_persistor.retrieve_tar.blob_found", blob_name=target_filename
            )
        except AzureError as exc:
            log = (
                f"An exception occurred while trying to download "
                f"the model '{target_filename}' in the specified container "
                f"'{self.container_name}'. Please make sure the model exists "
                f"in the container."
            )
            structlogger.error(
                "azure_persistor.retrieve_tar.model_download_error",
                container_name=self.container_name,
                target_filename=target_filename,
                event_info=log,
                exception=exc,
            )
            raise ModelNotFound("Model not found") from exc
