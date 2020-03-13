import io
import logging
import os
import shutil
import tarfile
from typing import List, Optional, Text, Tuple


logger = logging.getLogger(__name__)


def get_persistor(name: Text) -> Optional["Persistor"]:
    """Returns an instance of the requested persistor.

    Currently, `aws`, `gcs` and `azure` are supported"""

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

    return None


class Persistor:
    """Store models in cloud and fetch them when needed"""

    def persist(self, model_directory: Text, model_name: Text) -> None:
        """Uploads a model persisted in the `target_dir` to cloud storage."""

        if not os.path.isdir(model_directory):
            raise ValueError(f"Target directory '{model_directory}' not found.")

        file_key, tar_path = self._compress(model_directory, model_name)
        self._persist_tar(file_key, tar_path)

    def retrieve(self, model_name: Text, target_path: Text) -> None:
        """Downloads a model that has been persisted to cloud storage."""

        tar_name = model_name

        if not model_name.endswith("tar.gz"):
            # ensure backward compatibility
            tar_name = self._tar_name(model_name)

        self._retrieve_tar(tar_name)
        self._decompress(os.path.basename(tar_name), target_path)

    def list_models(self) -> List[Text]:
        """Lists all the trained models."""

        raise NotImplementedError

    def _retrieve_tar(self, filename: Text) -> Text:
        """Downloads a model previously persisted to cloud storage."""

        raise NotImplementedError("")

    def _persist_tar(self, filekey: Text, tarname: Text) -> None:
        """Uploads a model persisted in the `target_dir` to cloud storage."""

        raise NotImplementedError("")

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
    def _model_dir_and_model_from_filename(filename: Text) -> Tuple[Text, Text]:

        split = filename.split("___")
        if len(split) > 1:
            model_name = split[1].replace(".tar.gz", "")
            return split[0], model_name
        else:
            return split[0], ""

    @staticmethod
    def _tar_name(model_name: Text, include_extension: bool = True) -> Text:

        ext = ".tar.gz" if include_extension else ""
        return f"{model_name}{ext}"

    @staticmethod
    def _decompress(compressed_path: Text, target_path: Text) -> None:

        with tarfile.open(compressed_path, "r:gz") as tar:
            tar.extractall(target_path)  # target dir will be created if it not exists


class AWSPersistor(Persistor):
    """Store models on S3.

    Fetches them when needed, instead of storing them on the local disk."""

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

    def list_models(self) -> List[Text]:
        try:
            return [
                self._model_dir_and_model_from_filename(obj.key)[1]
                for obj in self.bucket.objects.filter()
            ]
        except Exception as e:
            logger.warning(f"Failed to list models in AWS. {e}")
            return []

    def _ensure_bucket_exists(
        self, bucket_name: Text, region_name: Optional[Text] = None
    ) -> None:
        import boto3
        import botocore

        if not region_name:
            region_name = boto3.DEFAULT_SESSION.region_name

        bucket_config = {"LocationConstraint": region_name}
        # noinspection PyUnresolvedReferences
        try:
            self.s3.create_bucket(
                Bucket=bucket_name, CreateBucketConfiguration=bucket_config
            )
        except botocore.exceptions.ClientError:
            pass  # bucket already exists

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

     Fetches them when needed, instead of storing them on the local disk."""

    def __init__(self, bucket_name: Text) -> None:
        from google.cloud import storage

        super().__init__()

        self.storage_client = storage.Client()
        self._ensure_bucket_exists(bucket_name)

        self.bucket_name = bucket_name
        self.bucket = self.storage_client.bucket(bucket_name)

    def list_models(self) -> List[Text]:

        try:
            blob_iterator = self.bucket.list_blobs()
            return [
                self._model_dir_and_model_from_filename(b.name)[1]
                for b in blob_iterator
            ]
        except Exception as e:
            logger.warning(f"Failed to list models in google cloud storage. {e}")
            return []

    def _ensure_bucket_exists(self, bucket_name: Text) -> None:
        from google.cloud import exceptions

        try:
            self.storage_client.create_bucket(bucket_name)
        except exceptions.Conflict:
            # bucket exists
            pass

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to GCS."""

        blob = self.bucket.blob(file_key)
        blob.upload_from_filename(tar_path)

    def _retrieve_tar(self, target_filename: Text) -> None:
        """Downloads a model that has previously been persisted to GCS."""

        blob = self.bucket.blob(target_filename)
        blob.download_to_filename(target_filename)


class AzurePersistor(Persistor):
    """Store models on Azure"""

    def __init__(
        self, azure_container: Text, azure_account_name: Text, azure_account_key: Text
    ) -> None:
        from azure.storage import blob as azureblob

        super().__init__()

        self.blob_client = azureblob.BlockBlobService(
            account_name=azure_account_name,
            account_key=azure_account_key,
            endpoint_suffix="core.windows.net",
        )

        self._ensure_container_exists(azure_container)
        self.container_name = azure_container

    def _ensure_container_exists(self, container_name: Text) -> None:

        exists = self.blob_client.exists(container_name)
        if not exists:
            self.blob_client.create_container(container_name)

    def list_models(self) -> List[Text]:

        try:
            blob_iterator = self.blob_client.list_blobs(self.container_name)
            return [
                self._model_dir_and_model_from_filename(b.name)[1]
                for b in blob_iterator
            ]
        except Exception as e:
            logger.warning(f"Failed to list models azure blob storage. {e}")
            return []

    def _persist_tar(self, file_key: Text, tar_path: Text) -> None:
        """Uploads a model persisted in the `target_dir` to Azure."""

        self.blob_client.create_blob_from_path(self.container_name, file_key, tar_path)

    def _retrieve_tar(self, target_filename: Text) -> None:
        """Downloads a model that has previously been persisted to Azure."""

        self.blob_client.get_blob_to_path(
            self.container_name, target_filename, target_filename
        )
