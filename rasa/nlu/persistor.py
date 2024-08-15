import abc
import structlog
import os
import shutil
from typing import Optional, Text, Tuple, TYPE_CHECKING

from rasa.shared.exceptions import RasaException

import rasa.shared.utils.common
import rasa.utils.common

if TYPE_CHECKING:
    from azure.storage.blob import ContainerClient

structlogger = structlog.get_logger()


def get_persistor(name: Text) -> Optional["Persistor"]:
    """Returns an instance of the requested persistor.

    Currently, `aws`, `gcs`, `azure` and providing module paths are supported remote
    storages.
    """
    if name == "aws":
        return AWSPersistor(
            os.environ.get("BUCKET_NAME"), os.environ.get("AWS_ENDPOINT_URL")
        )
    if name == "gcs":
        return GCSPersistor(os.environ.get("BUCKET_NAME"))

    if name == "azure":
        return AzurePersistor(
            os.environ.get("AZURE_CONTAINER"),
            os.environ.get("AZURE_ACCOUNT_NAME"),
            os.environ.get("AZURE_ACCOUNT_KEY"),
        )
    if name:
        try:
            persistor = rasa.shared.utils.common.class_from_module_path(name)
            return persistor()
        except ImportError:
            raise ImportError(
                f"Unknown model persistor {name}. Please make sure to "
                "either use an included model persistor (`aws`, `gcs` "
                "or `azure`) or specify the module path to an external "
                "model persistor."
            )
    return None


class Persistor(abc.ABC):
    """Store models in cloud and fetch them when needed."""

    def persist(self, trained_model: Text) -> None:
        """Uploads a trained model persisted in the `target_dir` to cloud storage."""
        file_key = os.path.basename(trained_model)
        self._persist_tar(file_key, trained_model)

    def retrieve(self, model_name: Text, target_path: Text) -> None:
        """Downloads a model that has been persisted to cloud storage."""
        tar_name = model_name
        if not model_name.endswith("tar.gz"):
            # ensure backward compatibility
            tar_name = self._tar_name(model_name)

        self._retrieve_tar(tar_name)
        self._copy(os.path.basename(tar_name), target_path)

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
        ext = ".tar.gz" if include_extension else ""
        return f"{model_name}{ext}"

    @staticmethod
    def _copy(compressed_path: Text, target_path: Text) -> None:
        shutil.copy2(compressed_path, target_path)


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
            if error_code == 403:
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
            elif error_code == 404:
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
